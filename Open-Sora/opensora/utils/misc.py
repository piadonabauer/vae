import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.cluster.dist_coordinator import DistCoordinator
from torch.utils.tensorboard import SummaryWriter

from opensora.acceleration.parallel_states import get_data_parallel_group

from .logger import log_message


def create_tensorboard_writer(exp_dir: str) -> SummaryWriter:
    """
    Create a tensorboard writer.

    Args:
        exp_dir (str): The directory to save tensorboard logs.

    Returns:
        SummaryWriter: The tensorboard writer.
    """
    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer


# ======================================================
# Memory
# ======================================================

GIGABYTE = 1024**3


def log_cuda_memory(stage: str = None):
    """
    Log the current CUDA memory usage.

    Args:
        stage (str): The stage of the training process.
    """
    text = "CUDA memory usage"
    if stage is not None:
        text += f" at {stage}"
    log_message(text + ": %.1f GB", torch.cuda.memory_allocated() / GIGABYTE)


def log_cuda_max_memory(stage: str = None):
    """
    Log the max CUDA memory usage.

    Args:
        stage (str): The stage of the training process.
    """
    torch.cuda.synchronize()
    max_memory_allocated = torch.cuda.max_memory_allocated()
    max_memory_reserved = torch.cuda.max_memory_reserved()
    log_message("CUDA max memory max memory allocated at " + stage + ": %.1f GB", max_memory_allocated / GIGABYTE)
    log_message("CUDA max memory max memory reserved at " + stage + ": %.1f GB", max_memory_reserved / GIGABYTE)


# ======================================================
# Number of parameters
# ======================================================


def get_model_numel(model: torch.nn.Module) -> tuple[int, int]:
    """
    Get the number of parameters in a model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        tuple[int, int]: The total number of parameters and the number of trainable parameters.
    """
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


def log_model_params(model: nn.Module):
    """
    Log the number of parameters in a model.

    Args:
        model (torch.nn.Module): The model.
    """
    num_params, num_params_trainable = get_model_numel(model)
    model_name = model.__class__.__name__
    log_message(f"[{model_name}] Number of parameters: {format_numel_str(num_params)}")
    log_message(f"[{model_name}] Number of trainable parameters: {format_numel_str(num_params_trainable)}")


def log_trainable_param_overview(
    model: nn.Module,
    label: str = "model",
    emit=None,
    detailed_lora_buckets: bool = False,
) -> None:
    """Log trainable vs frozen split, LoRA vs base, and a heuristic attention-related LoRA share.

    Args:
        emit: callable(str) -> None; default ``log_message``.
        detailed_lora_buckets: extra heuristic buckets (encoder/stem vs decoder vs viewwise).
    """
    emit = emit or log_message
    total = 0
    trainable = 0
    lora_t = 0
    lora_attn_t = 0
    lora_enc_t = 0
    lora_dec_t = 0
    lora_viewwise_t = 0
    attn_keywords = ("attn", "attention", "cross", "fusion", "mha", "qkv", "proj_attn", "to_q", "to_k", "to_v")
    enc_hints = ("encoder", "stem", "downsample", "conv1", "intro")
    dec_hints = ("decoder", "upsample", "up_", "mid.")
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if not p.requires_grad:
            continue
        trainable += n
        low = name.lower()
        if "lora" in low:
            lora_t += n
            if any(k in low for k in attn_keywords):
                lora_attn_t += n
            if detailed_lora_buckets:
                if "viewwise" in low or "view_wise" in low:
                    lora_viewwise_t += n
                elif any(h in low for h in enc_hints):
                    lora_enc_t += n
                elif any(h in low for h in dec_hints):
                    lora_dec_t += n
    if total == 0:
        return
    frozen = total - trainable
    base_t = trainable - lora_t
    lora_other_t = lora_t - lora_attn_t

    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    emit(
        f"[{label}] param overview: total={format_numel_str(total)} trainable={format_numel_str(trainable)} "
        f"({pct(trainable, total):.2f}% of weights) frozen={format_numel_str(frozen)}"
    )
    if lora_t > 0:
        emit(
            f"[{label}] LoRA trainable: {format_numel_str(lora_t)} ({pct(lora_t, trainable):.1f}% of trainable); "
            f"heuristic attention/cross-fusion LoRA: {format_numel_str(lora_attn_t)} "
            f"({pct(lora_attn_t, lora_t):.1f}% of LoRA); other LoRA: {format_numel_str(lora_other_t)}"
        )
        emit(
            f"[{label}] base trainable (non-LoRA): {format_numel_str(base_t)} ({pct(base_t, trainable):.1f}% of trainable)"
        )
        if detailed_lora_buckets:
            overlap_note = " (buckets overlap if a name matches multiple hints)"
            emit(
                f"[{label}] LoRA heuristic buckets{overlap_note}: "
                f"encoder/stem-ish={format_numel_str(lora_enc_t)}, "
                f"decoder-ish={format_numel_str(lora_dec_t)}, "
                f"viewwise-ish={format_numel_str(lora_viewwise_t)}"
            )
    else:
        emit(f"[{label}] no parameter names containing 'lora' among trainable weights (LoRA overview skipped)")


def log_wan_multiview_training_design_summary(model: nn.Module, model_cfg, emit=None) -> None:
    """One-time human-readable summary: fusion modes, LoRA switches, viewwise vs embeddings, rank, and param stats.

    ``model_cfg`` should be the training config's ``model`` dict (dict-like with ``.get``).
    """
    emit = emit or log_message
    m = model_cfg
    if not hasattr(m, "get"):
        m = dict(m)

    def ln(s: str) -> None:
        emit(s)

    ln("======== Wan multiview / crossview design summary (one-time) ========")

    use_cv = m.get("use_crossview_encoder", True)
    ln(f"use_crossview_encoder={use_cv}")
    if not use_cv:
        ln(
            "  -> Legacy path: WanVideoVAE + latent_fusion / latent_expand / view_group_fusion. "
            "Fields fusion_mode, use_lora_before/after, use_viewwise_decoder_lora apply to AttentionMultiViewVideoVan only; "
            "they do not affect this path."
        )
        ln("======== end design summary ========")
        log_trainable_param_overview(model, "VAE", emit=emit, detailed_lora_buckets=True)
        return

    fm = m.get("fusion_mode", "cross_attention")
    fusion_blurbs = {
        "cross_attention": (
            "Bidirectional cross-attention between views: each view attends to the other view’s tokens, "
            "then features are combined for the rest of the encoder/decoder. Strong when views are aligned but not identical."
        ),
        "self_attention": (
            "Concatenate tokens from all views into one sequence (length ≈ 2× spatial tokens), run self-attention, "
            "then concat channels + residual blocks. One joint mixing stage instead of pairwise cross-attn."
        ),
        "conv3d": (
            "Concatenate views along the channel axis and mix with 1×1×1 Conv3d + FusionResidualBlock3d. "
            "Cheaper than attention; inductive bias is local convolutional mixing rather than token attention."
        ),
    }
    ln(f"fusion_mode={fm!r}")
    ln(f"  -> {fusion_blurbs.get(fm, 'No built-in blurb for this mode; check AttentionMultiViewVideoVan in diffsynth.')}")

    ln(f"view_in={m.get('view_in')}, view_mixing_strategy={m.get('view_mixing_strategy', 'n/a')}")
    ln(
        "use_view_embedding vs use_viewwise_decoder_lora: "
        "per-view additive embeddings bias early (or latent) features with a small vector per view; "
        "viewwise_decoder_lora adds low-rank adapters on the decoder side conditioned by view index—typically more "
        "flexible than a single embedding per view for reconstruction, at the cost of more trainable parameters."
    )
    ln(f"  use_view_embedding={m.get('use_view_embedding', False)}")
    ln(f"  use_viewwise_decoder_lora={m.get('use_viewwise_decoder_lora', False)}")

    use_lora = m.get("use_lora", True)
    lb = m.get("use_lora_before", False)
    la = m.get("use_lora_after", True)
    r = m.get("lora_rank", 16)
    ln(
        f"use_lora={use_lora}, use_lora_before={lb}, use_lora_after={la}, lora_rank={r}. "
        "Before = LoRA on the encoder stem / early downsampling; after = bottleneck and later decoder. "
        "Both can be enabled for two complementary adaptation regions."
    )
    ln(
        "LoRA rank: all listed adapters in this stack typically share this rank. "
        "Larger rank increases adapter capacity and parameter count roughly linearly in rank; "
        "too high can overfit small datasets, too low can underfit cross-view cues."
    )

    if not use_lora:
        ln("  -> With use_lora=False, LoRA modules should be absent; trainable weights are mostly base conv/attn (subject to freeze_temporal/train_spatial).")

    if m.get("full_finetune_decoder", False):
        ln(
            "full_finetune_decoder=True: base conv/attn inside decoder LoRA wrappers are trainable "
            "alongside LoRA deltas; decode still uses per-view indices and nn.Embedding view conditioning "
            "in AttentionMultiViewVideoVan. For a heavier phase-2, consider use_lora=False to drop adapters entirely."
        )

    ln(f"freeze_temporal={m.get('freeze_temporal')}, train_spatial={m.get('train_spatial')}")
    ln("======== end design summary ========")

    log_trainable_param_overview(model, "VAE", emit=emit, detailed_lora_buckets=True)


# ======================================================
# String
# ======================================================


def format_numel_str(numel: int) -> str:
    """
    Format a number of elements to a human-readable string.

    Args:
        numel (int): The number of elements.

    Returns:
        str: The formatted string.
    """
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def format_duration(seconds: int) -> str:
    days, remainder = divmod(seconds, 86400)  # Extract days
    hours, remainder = divmod(remainder, 3600)  # Extract hours
    minutes, seconds = divmod(remainder, 60)  # Extract minutes and seconds

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:  # Always show seconds if nothing else
        parts.append(f"{seconds}s")

    return " ".join(parts)


# ======================================================
# PyTorch
# ======================================================


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, group=get_data_parallel_group())
    tensor.div_(dist.get_world_size(group=get_data_parallel_group()))
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, group=get_data_parallel_group())
    return tensor


def to_tensor(data: torch.Tensor | np.ndarray | Sequence | int | float) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: The converted tensor.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


def to_ndarray(data: torch.Tensor | np.ndarray | Sequence | int | float) -> np.ndarray:
    """Convert objects of various python types to :obj:`numpy.ndarray`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        numpy.ndarray: The converted ndarray.
    """
    if isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Sequence):
        return np.array(data)
    elif isinstance(data, int):
        return np.ndarray([data], dtype=int)
    elif isinstance(data, float):
        return np.array([data], dtype=float)
    else:
        raise TypeError(f"type {type(data)} cannot be converted to ndarray.")


def to_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """
    Convert a string or a torch.dtype to a torch.dtype.

    Args:
        dtype (str | torch.dtype): The input dtype.

    Returns:
        torch.dtype: The converted dtype.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError(f"Unsupported dtype {dtype}")
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


# ======================================================
# Profile
# ======================================================


class Timer:
    def __init__(self, name, log=False, barrier=False, coordinator: DistCoordinator | None = None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.log = log
        self.barrier = barrier
        self.coordinator = coordinator

    @property
    def elapsed_time(self) -> float:
        return self.end_time - self.start_time

    def __enter__(self):
        torch.cuda.synchronize()
        if self.barrier:
            dist.barrier()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.coordinator is not None:
            self.coordinator.block_all()
        torch.cuda.synchronize()
        if self.barrier:
            dist.barrier()
        self.end_time = time.time()
        if self.log:
            print(f"Elapsed time for {self.name}: {self.elapsed_time:.2f} s")


class Timers:
    def __init__(self, record_time: bool, record_barrier: bool = False, coordinator: DistCoordinator | None = None):
        self.timers = OrderedDict()
        self.record_time = record_time
        self.record_barrier = record_barrier
        self.coordinator = coordinator

    def __getitem__(self, name: str) -> Timer:
        if name not in self.timers:
            if self.record_time:
                self.timers[name] = Timer(name, barrier=self.record_barrier, coordinator=self.coordinator)
            else:
                self.timers[name] = nullcontext()
        return self.timers[name]

    def to_dict(self):
        return {f"time_debug/{name}": timer.elapsed_time for name, timer in self.timers.items()}

    def to_str(self, epoch: int, step: int) -> str:
        log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
        for name, timer in self.timers.items():
            log_str += f"{name}: {timer.elapsed_time:.2f} s | "
        return log_str


def is_pipeline_enabled(plugin_type: str, plugin_config: dict) -> bool:
    return plugin_type == "hybrid" and plugin_config.get("pp_size", 1) > 1


def is_log_process(plugin_type: str, plugin_config: dict) -> bool:
    if is_pipeline_enabled(plugin_type, plugin_config):
        return dist.get_rank() == dist.get_world_size() - 1
    return dist.get_rank() == 0


class NsysRange:
    def __init__(self, range_name: str):
        self.range_name = range_name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.range_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.nvtx.range_pop()


class NsysProfiler:
    """
    Use NVIDIA Nsight Systems to profile the code.

    Example (~30MB):
    ```bash
    /home/zhengzangwei/nsight-systems-2024.7.1/bin/nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown  -o cache/nsys/report2 \
        torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/stage2.py --nsys True --dataset.data-path /mnt/ddn/sora/meta/train/all_till_20241115_chunk901+img7.6M.parquet
    ```

    Example (~130MB + 2G):
    ```bash
    /home/zhengzangwei/nsight-systems-2024.7.1/bin/nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -s process-tree --cudabacktrace=all --stats=true -o cache/nsys/report5 \
        torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/stage2.py --nsys True --dataset.data-path /mnt/ddn/sora/meta/train/all_till_20241115_chunk901+img7.6M.parquet --record_time True --record_barrier True
    ```

    To generate summary statistics, use `--stats=true`.
    To disable stack traces, use use `-s none --cudabacktrace=none`.
    To use stack traces, use `-s process-tree --cudabacktrace=all`.
    To enable timer, use `--record_time True --record_barrier True` for `scripts/diffusion/train.py`.
    """

    def __init__(self, warmup_steps: int = 0, num_steps: int = 1, enabled: bool = True):
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.current_step = 0
        self.enabled = enabled

    def step(self):
        if not self.enabled:
            return
        self.current_step += 1
        if self.current_step == self.warmup_steps:
            torch.cuda.cudart().cudaProfilerStart()
        elif self.current_step >= self.warmup_steps + self.num_steps:
            torch.cuda.cudart().cudaProfilerStop()

    def range(self, range_name: str) -> NsysRange:
        if not self.enabled:
            return nullcontext()
        return NsysRange(range_name)


class ProfilerContext:
    def __init__(
        self,
        save_path: str = "./log",
        record_shapes: bool = False,
        with_stack: bool = True,
        wait: int = 1,
        warmup: int = 1,
        active: int = 1,
        repeat: int = 1,
        enable: bool = True,
        **kwargs,
    ):
        self.enable = enable
        self.prof = None
        self.step_cnt = 0
        self.total_steps = (wait + warmup + active) * repeat
        if enable:
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
                record_shapes=record_shapes,
                with_stack=with_stack,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(save_path),
                **kwargs,
            )

    def step(self):
        if self.enable:
            if self.step_cnt == 0:
                self.prof.__enter__()
            self.prof.step()
            self.step_cnt += 1
            if self.is_profile_end():
                self.prof.__exit__(None, None, None)
                exit(0)

    def is_profile_end(self):
        return self.step_cnt >= self.total_steps


def get_process_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3


def get_total_mem():
    return psutil.virtual_memory().used / 1024**3


def print_mem(prefix: str = ""):
    rank = dist.get_rank()
    print(
        f"[{rank}] {prefix} process memory: {get_process_mem():.2f} GB, total memory: {get_total_mem():.2f} GB",
        flush=True,
    )
