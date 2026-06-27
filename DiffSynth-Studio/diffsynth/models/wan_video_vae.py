from einops import rearrange, repeat

import json
import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

logger = logging.getLogger(__name__)

CACHE_T = 2


class ProfileTimer:
    """
    Lightweight CUDA-synchronized block timer for user-friendly profiling.

    Enable once per profiled step; call summarize() to get a readable report.
    """

    enabled = False
    records: dict[str, list[float]] = {}

    @classmethod
    def enable(cls, *, reset: bool = True) -> None:
        if reset:
            cls.records = {}
        cls.enabled = True

    @classmethod
    def disable(cls) -> None:
        cls.enabled = False

    @classmethod
    @contextmanager
    def block(cls, name: str):
        if not cls.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            cls.records.setdefault(name, []).append(ms)

    @classmethod
    def summarize(cls) -> dict:
        rows = []
        for name, times in cls.records.items():
            if not times:
                continue
            mean_ms = sum(times) / len(times)
            rows.append(
                {
                    "name": name,
                    "ms_mean": round(mean_ms, 3),
                    "ms_min": round(min(times), 3),
                    "ms_max": round(max(times), 3),
                    "ms_total": round(sum(times), 3),
                    "count": len(times),
                    "attention": name.startswith("attention."),
                }
            )
        rows.sort(key=lambda r: r["ms_mean"], reverse=True)
        return {
            "blocks": rows,
            "attention_blocks": [r for r in rows if r["attention"]],
            "other_blocks": [r for r in rows if not r["attention"]],
        }

    @classmethod
    def _row(cls, rows: list[dict], name: str) -> dict | None:
        for row in rows:
            if row["name"] == name:
                return row
        return None

    @classmethod
    def _pct(cls, part: float, whole: float) -> str:
        if whole <= 0:
            return "  n/a"
        return f"{100.0 * part / whole:5.1f}%"

    @classmethod
    def format_report(cls, summary: dict, *, step: int | None = None) -> str:
        rows = summary["blocks"]
        lines: list[str] = []

        title = "VAE block timing profile"
        if step is not None:
            title += f" (step {step})"
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)
        lines.append(
            "CUDA-synchronized timers. Nested blocks are INSIDE their parent "
            "(e.g. attention.view.sdpa is part of train.forward, not extra time)."
        )
        lines.append("")

        # ── Top-level step (non-overlapping) ─────────────────────────────────
        top_names = ["train.forward", "train.loss", "train.backward", "train.optimizer"]
        top_rows = [cls._row(rows, n) for n in top_names]
        top_rows = [r for r in top_rows if r is not None]
        step_total = sum(r["ms_mean"] for r in top_rows)

        lines.append("STEP BREAKDOWN (top-level, non-overlapping)")
        lines.append("-" * 42)
        lines.append(f"{'block':<28} {'mean_ms':>9} {'pct_step':>9}")
        for row in top_rows:
            lines.append(
                f"{row['name']:<28} {row['ms_mean']:>9.3f} {cls._pct(row['ms_mean'], step_total):>9}"
            )
        lines.append(f"{'STEP TOTAL (approx)':<28} {step_total:>9.3f}")
        lines.append("")

        # ── Forward breakdown (% of train.forward only) ───────────────────────
        fwd = cls._row(rows, "train.forward")
        if fwd:
            fwd_ms = fwd["ms_mean"]
            lines.append(f"FORWARD BREAKDOWN (% of train.forward = {fwd_ms:.1f} ms)")
            lines.append("-" * 42)
            lines.append(f"{'block':<28} {'mean_ms':>9} {'total_ms':>9} {'count':>6} {'pct_fwd':>9}")

            forward_blocks = [
                "encode.downsample_all_views",
                "encode.fusion.cross_attention",
                "encode.fusion.tree_merge",
                "encode.middle",
                "encode.head",
                "encode.latent_conv",
                "decode.view_condition",
                "decode.conv2",
                "decode.body.conv1",
                "decode.body.middle",
                "decode.body.upsamples",
                "decode.body.head",
                "decode.temporal_loop",
                "decode.decoder",  # legacy combined timer if still present
            ]
            accounted = 0.0
            for name in forward_blocks:
                row = cls._row(rows, name)
                if row is None:
                    continue
                # Use total_ms when block runs multiple times (per view / per frame).
                contrib = row["ms_total"] if row["count"] > 1 else row["ms_mean"]
                accounted += contrib
                lines.append(
                    f"{name:<28} {row['ms_mean']:>9.3f} {row['ms_total']:>9.3f} "
                    f"{row['count']:>6} {cls._pct(contrib, fwd_ms):>9}"
                )
            unaccounted = max(0.0, fwd_ms - accounted)
            lines.append(
                f"{'forward unaccounted':<28} {unaccounted:>9.3f} {'':>9} {'':>6} "
                f"{cls._pct(unaccounted, fwd_ms):>9}"
            )
            lines.append("")

        # ── Decoder detail (inside decode.temporal_loop / per-view) ───────────
        temporal = cls._row(rows, "decode.temporal_loop")
        all_views = cls._row(rows, "decode.all_views")
        view_rows = [r for r in rows if r["name"].startswith("decode.view_")]
        ups_rows = [r for r in rows if r["name"].startswith("decode.ups.L")]
        if temporal or all_views or view_rows or ups_rows:
            if all_views is not None:
                parent_ms = all_views["ms_total"]
                parent_label = "decode.all_views"
            elif temporal is not None:
                parent_ms = temporal["ms_total"]
                parent_label = "decode.temporal_loop"
            else:
                parent_ms = sum(r["ms_total"] for r in view_rows)
                parent_label = "decode views"
            lines.append(
                f"DECODER DETAIL (% of {parent_label} total = {parent_ms:.1f} ms)"
            )
            lines.append("-" * 42)
            lines.append(f"{'block':<32} {'mean_ms':>9} {'total_ms':>9} {'count':>6} {'pct_dec':>9}")

            decoder_detail = [
                "decode.all_views",
                "decode.view_0",
                "decode.view_1",
                "decode.view_2",
                "decode.view_3",
                "decode.temporal.decoder",
                "decode.temporal.cat",
                "decode.body.middle.resblocks",
                "decode.body.middle.attn",
            ]
            for name in decoder_detail:
                row = cls._row(rows, name)
                if row is None:
                    continue
                contrib = row["ms_total"] if row["count"] > 1 else row["ms_mean"]
                lines.append(
                    f"{name:<32} {row['ms_mean']:>9.3f} {row['ms_total']:>9.3f} "
                    f"{row['count']:>6} {cls._pct(contrib, parent_ms):>9}"
                )

            # Upsample stages: L0 = coarsest latent grid, L3 = finest before head.
            stage_ids = sorted(
                {name.split(".")[2] for name in (r["name"] for r in ups_rows)}
            )
            for stage in stage_ids:
                stage_rows = [r for r in ups_rows if f".{stage}." in r["name"]]
                stage_rows.sort(key=lambda r: r["name"])
                for row in stage_rows:
                    short = row["name"].split(".", 3)[-1]  # resblock / attn / resample
                    label = f"  {stage}.{short}"
                    lines.append(
                        f"{label:<32} {row['ms_mean']:>9.3f} {row['ms_total']:>9.3f} "
                        f"{row['count']:>6} {cls._pct(row['ms_total'], parent_ms):>9}"
                    )
            lines.append("")

        # ── View-attention fusion (inside cross_attention) ────────────────────
        view_attn = [r for r in rows if r["name"].startswith("attention.view.")]
        if view_attn:
            cross = cls._row(rows, "encode.fusion.cross_attention")
            parent_ms = cross["ms_mean"] if cross else sum(r["ms_mean"] for r in view_attn)
            lines.append(f"VIEW FUSION ATTENTION (inside encode.fusion.cross_attention, ~{parent_ms:.1f} ms)")
            lines.append("-" * 42)
            lines.append(f"{'block':<28} {'mean_ms':>9} {'pct_parent':>10}")
            for row in view_attn:
                lines.append(
                    f"{row['name']:<28} {row['ms_mean']:>9.3f} "
                    f"{cls._pct(row['ms_mean'], parent_ms):>10}"
                )
            lines.append("")

        # ── All AttentionBlock SDPA calls (encoder middle / decoder upsamples) ─
        block_sdpa = cls._row(rows, "attention.block.sdpa")
        if block_sdpa:
            lines.append("OTHER ATTENTION (AttentionBlock.sdpa in encoder/decoder stacks)")
            lines.append("-" * 42)
            lines.append(
                f"attention.block.sdpa   mean={block_sdpa['ms_mean']:.3f} ms/call  "
                f"total={block_sdpa['ms_total']:.3f} ms  count={block_sdpa['count']} calls"
            )
            if fwd:
                lines.append(
                    f"  → {cls._pct(block_sdpa['ms_total'], fwd['ms_mean'])} of train.forward"
                )
            lines.append("")

        # ── Raw dump for reference ────────────────────────────────────────────
        lines.append("ALL TIMED BLOCKS (raw, sorted by mean_ms)")
        lines.append("-" * 42)
        lines.append(f"{'block':<42} {'mean_ms':>9} {'total_ms':>9} {'count':>6}")
        for row in rows:
            lines.append(
                f"{row['name']:<42} {row['ms_mean']:>9.3f} {row['ms_total']:>9.3f} {row['count']:>6}"
            )

        return "\n".join(lines)

    @classmethod
    def save_report(cls, summary: dict, out_dir: str | Path, *, step: int | None = None) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_step{step}" if step is not None else ""
        json_path = out_dir / f"profile_timing{suffix}.json"
        txt_path = out_dir / f"profile_timing{suffix}.txt"
        payload = {"step": step, **summary}
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        txt_path.write_text(cls.format_report(summary, step=step), encoding="utf-8")
        return json_path, txt_path


def check_is_instance(model, module_class):
    if isinstance(model, module_class):
        return True
    if hasattr(model, "module") and isinstance(model.module, module_class):
        return True
    # LoRA-wrapped convolutions (e.g. LoRAConv3d) keep the original module in
    # ``base_conv`` and forward *args (incl. the temporal ``cache_x``) to it.
    # Treat them as instances of the wrapped type so feat_cache / temporal
    # compression flows through correctly even when use_lora_* wraps these convs.
    if hasattr(model, "base_conv") and isinstance(model.base_conv, module_class):
        return True
    return False


def block_causal_mask(x, block_size):
    # params
    b, n, s, _, device = *x.size(), x.device
    assert s % block_size == 0
    num_blocks = s // block_size

    # build mask
    mask = torch.zeros(b, n, s, s, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        mask[:, :,
             i * block_size:(i + 1) * block_size, :(i + 1) * block_size] = 1
    return mask


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim,
                                          dim * 2, (3, 1, 1),
                                          padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim,
                                          dim, (3, 1, 1),
                                          stride=(2, 1, 1),
                                          padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)



def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(x,
                      "b c f (h q) (w r) -> b (c r q) f h w",
                      q=patch_size,
                      r=patch_size)
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(x,
                      "b (c r q) f h w -> b c f (h q) (w r)",
                      q=patch_size,
                      r=patch_size)
    return x


class Resample38(Resample):

    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super(Resample, self).__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(
            0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        with ProfileTimer.block("attention.block.sdpa"):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                #attn_mask=block_causal_mask(q, block_size=h * w)
            )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample38(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False
    ):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample38(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout),
                                    AttentionBlock(out_dim),
                                    ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0], debug_shapes: bool = False, stage_name: str = "Encoder3d"):
        def _dbg(tag, tensor):
            if not debug_shapes:
                return
            if tensor is None:
                print(f"[{stage_name}] {tag}: None")
                return
            if tensor.dim() == 5:
                b, c, t, h, w = tensor.shape
                print(f"[{stage_name}] {tag}: B={b}, C={c}, T={t}, H={h}, W={w} | shape={tuple(tensor.shape)}")
            else:
                print(f"[{stage_name}] {tag}: shape={tuple(tensor.shape)}")

        _dbg("input", x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        _dbg("after initial CausalConv3d conv1", x)

        ## downsamples
        block_idx = 0
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
            if isinstance(layer, ResidualBlock):
                block_idx += 1
                _dbg(f"after ResidualBlock #{block_idx}", x)
            elif isinstance(layer, AttentionBlock):
                _dbg("after AttentionBlock (self-attn)", x)
            elif isinstance(layer, Resample):
                _dbg("after Downsample block (Resample)", x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
            if isinstance(layer, ResidualBlock):
                _dbg("middle ResidualBlock", x)
            elif isinstance(layer, AttentionBlock):
                _dbg("middle AttentionBlock (bottleneck attn)", x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
            if isinstance(layer, CausalConv3d):
                _dbg("after final CausalConv3d head", x)
        return x


class Encoder3d_38(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i] if i < len(temperal_downsample) else False
            )
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )


    def forward(self, x, feat_cache=None, feat_idx=[0]):

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0], debug_shapes: bool = False, stage_name: str = "Decoder3d"):
        def _dbg(tag, tensor):
            if not debug_shapes:
                return
            if tensor is None:
                print(f"[{stage_name}] {tag}: None")
                return
            if tensor.dim() == 5:
                b, c, t, h, w = tensor.shape
                print(f"[{stage_name}] {tag}: B={b}, C={c}, T={t}, H={h}, W={w} | shape={tuple(tensor.shape)}")
            else:
                print(f"[{stage_name}] {tag}: shape={tuple(tensor.shape)}")

        _dbg("input_z", x)
        ## conv1
        with ProfileTimer.block("decode.body.conv1"):
            if feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = self.conv1(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = self.conv1(x)
        _dbg("after initial CausalConv3d conv1", x)

        ## middle
        with ProfileTimer.block("decode.body.middle"):
            for layer in self.middle:
                block_name = (
                    "decode.body.middle.attn"
                    if isinstance(layer, AttentionBlock)
                    else "decode.body.middle.resblocks"
                )
                with ProfileTimer.block(block_name):
                    if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                        x = layer(x, feat_cache, feat_idx)
                    else:
                        x = layer(x)
                if isinstance(layer, ResidualBlock):
                    _dbg("middle ResidualBlock", x)
                elif isinstance(layer, AttentionBlock):
                    _dbg("middle AttentionBlock (bottleneck attn)", x)

        ## upsamples — per resolution stage (L0 = coarsest, +resample → L1, …)
        stage_idx = 0
        with ProfileTimer.block("decode.body.upsamples"):
            block_idx = 0
            for layer in self.upsamples:
                if isinstance(layer, Resample):
                    block_name = f"decode.ups.L{stage_idx}.resample"
                    stage_idx += 1
                elif isinstance(layer, AttentionBlock):
                    block_name = f"decode.ups.L{stage_idx}.attn"
                else:
                    block_name = f"decode.ups.L{stage_idx}.resblock"
                with ProfileTimer.block(block_name):
                    if feat_cache is not None:
                        x = layer(x, feat_cache, feat_idx)
                    else:
                        x = layer(x)
                if isinstance(layer, ResidualBlock):
                    block_idx += 1
                    _dbg(f"after Upsample ResidualBlock #{block_idx}", x)
                elif isinstance(layer, AttentionBlock):
                    _dbg("after Upsample AttentionBlock", x)
                elif isinstance(layer, Resample):
                    _dbg("after Upsample block (Resample)", x)

        ## head
        with ProfileTimer.block("decode.body.head"):
            for layer in self.head:
                if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                    idx = feat_idx[0]
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)
                    x = layer(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    x = layer(x)
                if isinstance(layer, CausalConv3d):
                    _dbg("after final CausalConv3d head", x)
        return x



class Decoder3d_38(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(in_dim=in_dim,
                                 out_dim=out_dim,
                                 dropout=dropout,
                                 mult=num_res_blocks + 1,
                                 temperal_upsample=t_up_flag,
                                 up_flag=i != len(dim_mult) - 1))
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, 12, 3, padding=1))


    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count

class VideoVAE_(nn.Module):

    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # Debug flag: when True, print step-by-step tensor shapes in encode/decode.
        self.debug_shapes = False

        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout
        )

    def _print_shape(self, tag, x):
        if not self.debug_shapes:
            return
        if x is None:
            print(f"[VideoVAE_] {tag}: None")
            return
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            print(f"[VideoVAE_] {tag}: B={b}, C={c}, T={t}, H={h}, W={w} | shape={tuple(x.shape)}")
        else:
            print(f"[VideoVAE_] {tag}: shape={tuple(x.shape)}")

    def forward(self, x, debug_shapes: bool | None = None):
        """
        Forward pass with optional shape debugging.

        Args:
            x: input video [B, C, T, H, W]
            debug_shapes: if True, prints shapes after each major step.
                          If None, uses self.debug_shapes.
        """
        if debug_shapes is not None:
            self.debug_shapes = debug_shapes

        #self._print_shape("input (x)", x)
        mu, log_var = self.encode(x, self.scale if hasattr(self, "scale") else [0.0, 1.0])
        #self._print_shape("latent mu", mu)
        #self._print_shape("latent log_var", log_var)

        z = self.reparameterize(mu, log_var)
        #self._print_shape("sampled z", z)

        x_recon = self.decode(z, self.scale if hasattr(self, "scale") else [0.0, 1.0])
        #self._print_shape("reconstruction (x_recon)", x_recon)

        return x_recon, mu, log_var

    def _encode_with_stats(self, x, scale):
        self.clear_cache()
        #self._print_shape("encode/input", x)

        # Time chunking for long sequences
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                x_chunk = x[:, :, :1, :, :]
                #self._print_shape(f"encode/chunk_{i}_input", x_chunk)
                out = self.encoder(
                    x_chunk,
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    debug_shapes=self.debug_shapes,
                    stage_name="Encoder3d",
                )
                #self._print_shape(f"encode/chunk_{i}_output_after_encoder", out)
            else:
                x_chunk = x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]
                #self._print_shape(f"encode/chunk_{i}_input", x_chunk)
                out_ = self.encoder(
                    x_chunk,
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    debug_shapes=self.debug_shapes,
                    stage_name="Encoder3d",
                )
                #self._print_shape(f"encode/chunk_{i}_output_after_encoder", out_)
                out = torch.cat([out, out_], 2)
                #self._print_shape(f"encode/concat_after_chunk_{i}", out)

        out_conv = self.conv1(out)
        #self._print_shape("encode/after_conv1", out_conv)
        mu, log_var = out_conv.chunk(2, dim=1)
        #self._print_shape("encode/mu_raw", mu)
        #self._print_shape("encode/log_var_raw", log_var)

        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        #self._print_shape("encode/mu_scaled", mu)
        return mu, log_var

    def encode(self, x, scale):
        """
        Public encode API used by Wan wrappers.
        Returns only the latent mu tensor for compatibility.
        """
        mu, _ = self._encode_with_stats(x, scale)
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        #self._print_shape("decode/input_z", z)

        # z: [B, C, T, H, W]
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        #self._print_shape("decode/z_rescaled", z)

        iter_ = z.shape[2]
        x = self.conv2(z)
        #self._print_shape("decode/after_conv2", x)

        out = None
        for i in range(iter_):
            self._conv_idx = [0]
            x_chunk = x[:, :, i : i + 1, :, :]
            #self._print_shape(f"decode/chunk_{i}_input", x_chunk)
            if i == 0:
                out = self.decoder(
                    x_chunk,
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    debug_shapes=self.debug_shapes,
                    stage_name="Decoder3d",
                )
                #self._print_shape(f"decode/chunk_{i}_output_after_decoder", out)
            else:
                out_ = self.decoder(
                    x_chunk,
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    debug_shapes=self.debug_shapes,
                    stage_name="Decoder3d",
                )
                #self._print_shape(f"decode/chunk_{i}_output_after_decoder", out_)
                out = torch.cat([out, out_], 2)  # may add tensor offload
                #self._print_shape(f"decode/concat_after_chunk_{i}", out)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self._encode_with_stats(imgs, self.scale if hasattr(self, "scale") else [0.0, 1.0])
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


class ViewPositionalEmbedding(nn.Module):
    """
    Learnable view embeddings to encode camera perspective into latent space.

    Why: Provides the model a way to distinguish which camera view is being reconstructed.
    
    KEY INSIGHT: Embeddings alone can't recover information lost in averaging.
    They work best when views are NOT compressed/averaged first.
    
    Use with view_compression=1 for best results (no compression before embeddings).
    """

    def __init__(self, num_views, channels, use_multiplicative=True):
        super().__init__()
        self.num_views = num_views
        self.channels = channels
        self.use_multiplicative = use_multiplicative
        
        # Additive embedding: Small learnable shifts per view
        # Keep magnitude small so we don't corrupt latent space
        self.embedding_add = nn.Parameter(torch.randn(1, num_views, channels, 1, 1, 1) * 0.05)
        
        # Multiplicative embedding: Small learnable scale per view
        if use_multiplicative:
            # Initialize near 1.0 so scaling is gentle (0.95 to 1.05 range)
            mul_init = torch.ones(1, num_views, channels, 1, 1, 1)
            mul_init = mul_init + torch.randn(1, num_views, channels, 1, 1, 1) * 0.05
            self.embedding_mul = nn.Parameter(mul_init)
        else:
            self.embedding_mul = None

    def forward(self, x):
        # x: [B, V, C, T, H, W]
        # Apply per-view modulation: gentle scaling + shifting
        # Magnitude designed to NOT corrupt the learned latent distribution
        if self.embedding_mul is not None:
            return x * self.embedding_mul + self.embedding_add
        else:
            return x + self.embedding_add

class ViewCompressor(nn.Module):
    """
    Lightweight view mixing layer to compress or expand the view axis.

    Why: we want a single 4D latent space that captures inter-view redundancy
    without rewriting the entire 3D VAE stack. This projects the view axis
    with a 1x1 conv (linear layer) while preserving spatial/temporal structure.
    
    IMPROVED: Uses selective/attentive pooling instead of simple averaging to
    better preserve view-specific features.
    """

    def __init__(self, in_views, out_views, init="avg", use_learned_weights=True):
        super().__init__()
        self.in_views = int(in_views)
        self.out_views = int(out_views)
        self.use_learned_weights = use_learned_weights
        
        if self.in_views == self.out_views:
            # No-op path keeps parameters stable when no compression is needed.
            self.proj = nn.Identity()
            self.learned_weights = None
        else:
            # Conv1d over the view axis: this is the "linear layer" for view mixing.
            self.proj = nn.Conv1d(self.in_views, self.out_views, kernel_size=1, bias=True)
            
            # Optional: learned attention weights for view pooling
            if use_learned_weights:
                # Attention weights: [out_views, in_views]
                # Each output view learns which input views to focus on
                self.learned_weights = nn.Parameter(torch.ones(self.out_views, self.in_views))
                nn.init.xavier_uniform_(self.learned_weights)
            else:
                self.learned_weights = None
            
            if init == "avg":
                # Start from an average across views to keep recon stable at init.
                with torch.no_grad():
                    self.proj.weight.fill_(1.0 / max(1, self.in_views))
                    nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: [B, V, C, T, H, W] -> [B, V', C, T, H, W]
        # We permute so the view axis becomes the Conv1d "channel".
        if isinstance(self.proj, nn.Identity):
            return x
        
        b, v, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 5, 1).contiguous()  # B C T H W V
        x = x.view(-1, v)  # (B*C*T*H*W) x V
        
        if self.use_learned_weights and self.learned_weights is not None:
            # Apply learned attention-based mixing
            # weights: [V_out, V_in], x: [N, V_in]
            weights = torch.softmax(self.learned_weights, dim=1)  # Normalize
            x = torch.matmul(x, weights.t())  # [N, V_in] @ [V_in, V_out] -> [N, V_out]
            v_out = weights.shape[0]
        else:
            # Standard learned projection
            x = x.unsqueeze(-1)  # [N, V, 1] for Conv1d
            x = self.proj(x)  # [N, V_out, 1]
            x = x.squeeze(-1)  # [N, V_out]
            v_out = self.out_views
        
        x = x.view(b, c, t, h, w, v_out).permute(0, 5, 1, 2, 3, 4).contiguous()
        return x


class MultiViewVideoVAE_(nn.Module):
    """
    Multi-view wrapper over the Wan 3D VAE.

    What: encodes each view with the shared 3D VAE, then mixes/compresses the
    view axis in latent space. Decoding expands back to per-view latents.
    Why: minimal architectural change while enabling a 4D latent (time+view).
    """

    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        view_in=8, # just default, uses actual view_in from input
        view_out=2, # just default, uses actual view_out from input
        use_view_embedding=True,
        view_init="avg",
    ):
        super().__init__()
        self.view_in = int(view_in)
        self.view_out = int(view_out)
        self.base = VideoVAE_(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
        )
        self.view_embed = (
            ViewPositionalEmbedding(self.view_in, z_dim) if use_view_embedding else nn.Identity()
        )
        self.view_compress = ViewCompressor(self.view_in, self.view_out, init=view_init)
        self.view_expand = ViewCompressor(self.view_out, self.view_in, init=view_init)

    def encode(self, x, scale):
        # x: [B, V, C, T, H, W] -> z: [B, Vc, Cz, T', H', W']
        # Keep the base 3D VAE intact; view mixing happens after encoding.
        b, v, c, t, h, w = x.shape
        if v != self.view_in:
            raise ValueError(f"Expected {self.view_in} views, got {v}")
        x = x.view(b * v, c, t, h, w)
        z = self.base.encode(x, scale)
        z = z.view(b, v, self.base.z_dim, z.shape[2], z.shape[3], z.shape[4])
        z = self.view_embed(z)
        z = self.view_compress(z)
        return z

    def decode(self, z, scale):
        # z: [B, Vc, Cz, T', H', W'] -> x: [B, V, 3, T, H, W]
        # Expand view axis before decoding per-view frames.
        z = self.view_expand(z)
        b, v, c, t, h, w = z.shape
        z = z.view(b * v, c, t, h, w)
        x = self.base.decode(z, scale)
        x = x.view(b, v, 3, x.shape[2], x.shape[3], x.shape[4])
        return x



# NEW
class ViewAttention(nn.Module):
    """
    Self-attention over all views' spatial tokens (per temporal step).

    Multi-head: the channel dim is split into ``num_heads`` so the per-head
    dimension is small (~64). This keeps SDPA on the Flash / memory-efficient
    backends (head_dim <= 256), which is O(seq) memory instead of materializing
    the (V*N) x (V*N) score matrix — critical at high resolution where
    V*N grows quadratically (512px: V=4, N=64*64 -> 16384 tokens per step).
    """

    def __init__(self, dim, num_heads: int | None = None):
        super().__init__()
        self.dim = dim
        # Target head_dim ~64 (Flash-friendly). Fall back to a divisor of dim.
        if num_heads is None:
            num_heads = max(1, dim // 64)
            while dim % num_heads != 0:
                num_heads -= 1
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params so the block is identity at init
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        identity = x
        b, t, v, n, c = x.shape
        H, hd = self.num_heads, self.head_dim
        s = v * n  # sequence length (all views' spatial tokens)

        with ProfileTimer.block("attention.view.qkv_proj"):
            # [B, T, V, N, C] -> [B*T, C, V, N]
            x = rearrange(x, 'b t v n c -> (b t) c v n')
            x = self.norm(x)
            qkv = self.to_qkv(x).reshape(b * t, 3 * c, s)  # [B*T, 3C, S]
            q, k, val = qkv.chunk(3, dim=1)  # each [B*T, C, S]

            def to_heads(z):
                # [B*T, C, S] -> [B*T, H, S, hd]
                return z.reshape(b * t, H, hd, s).permute(0, 1, 3, 2).contiguous()

            q, k, val = to_heads(q), to_heads(k), to_heads(val)

        with ProfileTimer.block("attention.view.sdpa"):
            x = F.scaled_dot_product_attention(q, k, val)  # [B*T, H, S, hd]

        with ProfileTimer.block("attention.view.out_proj"):
            # [B*T, H, S, hd] -> [B*T, C, V, N]
            x = x.permute(0, 1, 3, 2).reshape(b * t, c, v, n)
            x = self.proj(x)

            # [B*T, C, V, N] -> [B, T, V, N, C]
            x = rearrange(x, '(b t) c v n -> b t v n c', t=t)
        return x + identity


class JointViewAttention(nn.Module):
    """
    Self-attention over both views' spatial tokens per time step.

    Concatenates view tokens to length 2*N at each t, runs MHA, and splits back.
    """

    def __init__(self, embed_dim=384, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, tokens_per_view):
        # tokens_per_view: list[[B, T, N, C]] with arbitrary number of views.
        if len(tokens_per_view) < 2:
            return tokens_per_view
        b, t, n, c = tokens_per_view[0].shape
        with ProfileTimer.block("attention.joint.preprocess"):
            x = torch.cat(tokens_per_view, dim=2)  # [B, T, V*N, C]
            x_norm = self.norm(x)
            x_norm_bt = x_norm.reshape(b * t, x.shape[2], c)
        with ProfileTimer.block("attention.joint.mha"):
            attn_out, _ = self.attn(x_norm_bt, x_norm_bt, x_norm_bt, need_weights=False)
        with ProfileTimer.block("attention.joint.postprocess"):
            attn_out = attn_out.reshape(b, t, x.shape[2], c)
            x_enriched = x + attn_out
            return list(x_enriched.split(n, dim=2))


class LoRAConv3d(nn.Module):
    """
    LoRA adapter for 3D convolutions using 1x1x1 low-rank updates.

    The base convolution is kept frozen; only the low-rank path is trainable.
    """

    def __init__(self, base_conv: CausalConv3d | nn.Conv3d, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base_conv, (CausalConv3d, nn.Conv3d))
        self.base_conv = base_conv
        for p in self.base_conv.parameters():
            p.requires_grad = False

        in_channels = base_conv.in_channels
        out_channels = base_conv.out_channels
        self.rank = rank
        self.alpha = alpha

        # Low-rank 1x1x1 convs
        self.lora_down = nn.Conv3d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv3d(rank, out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x, *args, **kwargs):
        # Preserve any extra args (e.g. cache) for CausalConv3d
        base_out = self.base_conv(x, *args, **kwargs) if isinstance(
            self.base_conv, CausalConv3d
        ) else self.base_conv(x)
        lora_out = self.lora_up(self.lora_down(x)) * self.alpha
        return base_out + lora_out


class LoRAConv2d(nn.Module):
    """
    LoRA adapter for 2D convolutions (e.g. spatial downsample in ``Resample``).

    Base conv is frozen. Low-rank path: 1×1 down → r→r conv with **same** kernel,
    stride, padding, dilation as ``base_conv`` → 1×1 up, so outputs match ``base_conv(x)``.
    """

    def __init__(self, base_conv: nn.Conv2d, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        self.base_conv = base_conv
        for p in self.base_conv.parameters():
            p.requires_grad = False

        in_channels = base_conv.in_channels
        out_channels = base_conv.out_channels
        self.rank = rank
        self.alpha = alpha

        ks = base_conv.kernel_size
        st = base_conv.stride
        pad = base_conv.padding
        dil = base_conv.dilation
        if isinstance(ks, int):
            ks = (ks, ks)
        if isinstance(st, int):
            st = (st, st)
        if isinstance(pad, int):
            pad = (pad, pad)
        if isinstance(dil, int):
            dil = (dil, dil)

        if base_conv.groups != 1:
            raise NotImplementedError(
                "LoRAConv2d for grouped Conv2d is not implemented; use groups=1."
            )

        self.lora_down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_mid = nn.Conv2d(
            rank,
            rank,
            kernel_size=ks,
            stride=st,
            padding=pad,
            dilation=dil,
            bias=False,
        )
        self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        h = self.lora_up(self.lora_mid(self.lora_down(x)))
        return self.base_conv(x) + self.alpha * h


class LoRAAttentionBlock(nn.Module):
    """
    AttentionBlock with LoRA adapters on Q, K, V and output projections.

    Copies weights from a pre-initialized AttentionBlock instance.
    """

    def __init__(self, base_attn: AttentionBlock, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base_attn, AttentionBlock)
        dim = base_attn.dim
        self.dim = dim

        # Copy RMSNorm
        self.norm = base_attn.norm

        # Freeze base convs
        self.to_qkv = base_attn.to_qkv
        self.proj = base_attn.proj
        for p in self.to_qkv.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = False

        # LoRA for qkv and proj (implemented as additional 1x1 convs on channels)
        self.rank = rank
        self.alpha = alpha

        # qkv: Conv2d(dim, 3*dim, 1)
        self.lora_qkv_down = nn.Conv2d(dim, rank, kernel_size=1, bias=False)
        self.lora_qkv_up = nn.Conv2d(rank, 3 * dim, kernel_size=1, bias=False)

        # proj: Conv2d(dim, dim, 1)
        self.lora_proj_down = nn.Conv2d(dim, rank, kernel_size=1, bias=False)
        self.lora_proj_up = nn.Conv2d(rank, dim, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_qkv_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_qkv_up.weight)
        nn.init.kaiming_uniform_(self.lora_proj_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_proj_up.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)

        # Base qkv
        base_qkv = self.to_qkv(x)
        # LoRA qkv
        lora_qkv = self.lora_qkv_up(self.lora_qkv_down(x)) * self.alpha
        qkv = base_qkv + lora_qkv

        q, k, v = (
            qkv.reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        base_proj = self.proj(x)
        lora_proj = self.lora_proj_up(self.lora_proj_down(x)) * self.alpha
        x = base_proj + lora_proj

        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


class FusionResidualBlock3d(nn.Module):
    """
    Same layout as ResidualBlock (norm → SiLU → conv → …) but with symmetric nn.Conv3d
    (non-causal) for view-fusion stacks where full temporal context is desired.
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = nn.Conv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        del feat_cache, feat_idx
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer(x)
        return x + h


class AttentionMultiViewVideoVan(nn.Module):
    """
    Two-view 3D VAE with encoder-side cross-view fusion and optional LoRA.

    - Input:  [B, 2, 3, T, H, W]
    - Output: fused latent [B, z_dim, T', H', W'] (same as VideoVAE_)

    LoRA placement:
    - ``use_lora_before``: Encoder3d stem *before* fusion (``encoder.conv1`` + ``encoder.downsamples``),
      i.e. the per-view path. Additive with ``use_lora_after``.
    - ``use_lora_after``: ``encoder.middle``, ``encoder.head``, and full ``decoder`` (post-fusion).
    """

    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        use_lora: bool = True,
        lora_rank: int = 16,
        fusion_mode: str = "cross_attention",
        use_lora_before: bool = False,
        use_lora_after: bool = True,
        use_viewwise_decoder_lora: bool = False,
        num_views: int = 2,
        temporal_compression: bool = True,
        grad_checkpoint: bool = False,
        view_attn_num_heads: int | None = None,
    ):
        super().__init__()
        # Back-compat: old name "joint_attention" is the same as "self_attention".
        if fusion_mode == "joint_attention":
            fusion_mode = "self_attention"
        # When True, run the per-view encoder/decoder with the chunked feat_cache
        # mechanism so Wan's temporal stride-convs actually fire (4x temporal
        # compression: T -> 1 + (T-1)//4). When False, T is preserved (legacy).
        self.temporal_compression = temporal_compression
        # Activation checkpointing for the heavy per-view down-path (encode) and
        # per-view decoder (decode). Trades ~20-30% compute for a large memory cut.
        self.grad_checkpoint = grad_checkpoint
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.fusion_mode = fusion_mode
        self.use_lora_before = use_lora_before
        self.use_lora_after = use_lora_after
        self.use_viewwise_decoder_lora = use_viewwise_decoder_lora
        self.num_views = int(num_views)
        self._encode_call_count = 0
        self._encode_logged = False  # used instead of _encode_call_count to avoid torch.compile recompilation

        # Reuse the standard encoder/decoder stacks
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout
        )

        # Learnable per-view embeddings for view-conditioned decoding.
        # VIEW_EMBEDDINGS 1: table is created with num_views rows and z_dim columns
        self.view_embed = nn.Embedding(self.num_views, z_dim) # eg num_views=4, latent dim for Wan=16

        bottleneck_channels = dim * dim_mult[-1]
        fused_channels = bottleneck_channels * self.num_views  # concatenated along channels

        # Fusion modules at feature level (after downsamples, before bottleneck middle/head).
        self.cross_attn = None
        self.joint_attn = None
        self.view_conv_fuse = None
        self.view_conv_norm = None
        self.view_conv_act = None
        self.conv4d_spatial = None
        self.conv4d_temporal = None
        self.conv4d_view = None
        self.conv4d_norm = None
        self.conv4d_act = None
        self.fusion_resblock1 = None
        self.fusion_resblock2 = None
        self.tree_resblocks = None  # per-level pairs for cross_attention tree merge

        if fusion_mode == "cross_attention":
            self.cross_attn = ViewAttention(dim=bottleneck_channels, num_heads=view_attn_num_heads)
            # Hierarchical pairwise tree merge with an INDEPENDENT (rb1, rb2) pair per MERGE
            # (i.e. per loop iteration), NOT shared per tree level.
            # For the carry-forward binary tree the number of pairwise merges is exactly
            # num_views - 1:
            #   4 views -> (v0,v1), (v2,v3), (v01,v23)            = 3 merges
            #   8 views -> 4 + 2 + 1                              = 7 merges
            #   3 views -> (v0,v1), (v01,v2)                      = 2 merges
            # Each merge sees features with different semantics (raw per-view early, partially
            # fused later), so giving every merge its own weights maximizes specialization.
            self.num_tree_merges = max(1, self.num_views - 1)
            self.tree_resblocks = nn.ModuleList([
                nn.ModuleList([
                    ResidualBlock(2 * bottleneck_channels, bottleneck_channels, dropout),
                    ResidualBlock(bottleneck_channels, bottleneck_channels, dropout),
                ])
                for _ in range(self.num_tree_merges)
            ])
            """
            # Per-level mean-pool init for rb1 at each level:
            # (shortcut: output[c] = 0.5*view_a[c] + 0.5*view_b[c]; residual path ≈ 0).
            C = bottleneck_channels
            for _rb1, _rb2 in self.tree_resblocks:
                _sc = _rb1.shortcut  # CausalConv3d(2C, C, 1)
                nn.init.zeros_(_sc.weight)
                for _c in range(C):
                    _sc.weight.data[_c, _c, 0, 0, 0] = 0.5       # from view a
                    _sc.weight.data[_c, _c + C, 0, 0, 0] = 0.5   # from view b
                if _sc.bias is not None:
                    nn.init.zeros_(_sc.bias)
                for _rb in (_rb1, _rb2):
                    _last = _rb.residual[-1]
                    nn.init.zeros_(_last.weight)
                    if _last.bias is not None:
                        nn.init.zeros_(_last.bias)
            """
        elif fusion_mode == "self_attention":
            # Full-sequence self-attention over both views' tokens (JointViewAttention), then concat + ResBlocks.
            self.joint_attn = JointViewAttention(embed_dim=bottleneck_channels, num_heads=8)
            self.fusion_resblock1 = ResidualBlock(fused_channels, bottleneck_channels, dropout)
            self.fusion_resblock2 = ResidualBlock(bottleneck_channels, bottleneck_channels, dropout)
        elif fusion_mode == "conv3d":
            # 1×1×1 Conv3d → GroupNorm + SiLU → two FusionResidualBlock3d (symmetric Conv3d).
            self.view_conv_fuse = nn.Conv3d(fused_channels, bottleneck_channels, kernel_size=1)
            gn_groups = 32 if bottleneck_channels % 32 == 0 else 16
            self.view_conv_norm = nn.GroupNorm(gn_groups, bottleneck_channels)
            self.view_conv_act = nn.SiLU()
            self.fusion_resblock1 = FusionResidualBlock3d(bottleneck_channels, bottleneck_channels, dropout)
            self.fusion_resblock2 = FusionResidualBlock3d(bottleneck_channels, bottleneck_channels, dropout)
        elif fusion_mode == "conv4d":
            # Factorized 4D fusion:
            # (B,C,T,V,H,W) -> spatial Conv2d over (H,W), then temporal Conv3d over (T,H,W),
            # then view Conv3d over (V,H,W) to compress V -> 1.
            self.conv4d_spatial = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
            self.conv4d_temporal = nn.Conv3d(
                bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1
            )
            self.conv4d_view = nn.Conv3d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=(self.num_views, 3, 3),
                padding=(0, 1, 1),
            )
            gn_groups = 32 if bottleneck_channels % 32 == 0 else 16
            self.conv4d_norm = nn.GroupNorm(gn_groups, bottleneck_channels)
            self.conv4d_act = nn.SiLU()
            self.fusion_resblock1 = FusionResidualBlock3d(bottleneck_channels, bottleneck_channels, dropout)
            self.fusion_resblock2 = FusionResidualBlock3d(bottleneck_channels, bottleneck_channels, dropout)
        else:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode}. "
                "Use one of: cross_attention, self_attention, conv3d, conv4d."
            )

        # Optional per-view latent LoRA adapters for decoding.
        # These replace additive view embeddings when enabled.
        if self.use_viewwise_decoder_lora:
            self.view_lora_down = nn.ModuleList(
                [nn.Conv3d(z_dim, lora_rank, kernel_size=1, bias=False) for _ in range(self.num_views)]
            )
            self.view_lora_up = nn.ModuleList(
                [nn.Conv3d(lora_rank, z_dim, kernel_size=1, bias=False) for _ in range(self.num_views)]
            )
            for i in range(self.num_views):
                nn.init.kaiming_uniform_(self.view_lora_down[i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.view_lora_up[i].weight)
        else:
            self.view_lora_down = None
            self.view_lora_up = None

        # Optional: wrap middle + decoder with LoRA
        if self.use_lora:
            self._enable_lora()

    def _enable_lora(self):
        # Helpers to recursively wrap conv and attention modules with LoRA
        def wrap_conv_with_lora(module: nn.Module):
            """Wrap CausalConv3d / Conv3d / Conv2d in the subtree (incl. Resample spatial convs)."""
            for name, child in list(module.named_children()):
                if isinstance(child, (CausalConv3d, nn.Conv3d)):
                    # LoRAConv3d adds a stride-1 1x1x1 parallel path, which only matches the base
                    # conv's output shape when the base preserves spatial/temporal size. Temporally
                    # (or spatially) strided 3D convs -- e.g. the Resample downsample3d time_conv with
                    # stride=(2,1,1) -- downsample T, so base_out and lora_out shapes differ and
                    # "base_out + lora_out" crashes. Leave those frozen (no LoRA); LoRAConv2d already
                    # handles strided 2D spatial downsamples correctly.
                    if any(s != 1 for s in child.stride):
                        continue
                    setattr(
                        module,
                        name,
                        LoRAConv3d(child, rank=self.lora_rank, alpha=1.0),
                    )
                elif isinstance(child, nn.Conv2d):
                    setattr(
                        module,
                        name,
                        LoRAConv2d(child, rank=self.lora_rank, alpha=1.0),
                    )
                else:
                    wrap_conv_with_lora(child)

        def wrap_attn_with_lora(module: nn.Module):
            for name, child in list(module.named_children()):
                if isinstance(child, AttentionBlock):
                    setattr(
                        module,
                        name,
                        LoRAAttentionBlock(child, rank=self.lora_rank, alpha=1.0),
                    )
                else:
                    wrap_attn_with_lora(child)

        # "Before" LoRA: shared Encoder3d stem used *before* view fusion (conv1 + downsamples per view).
        # This is additive to "after" (middle/head/decoder), not "LoRA on fusion blocks".
        if self.use_lora_before:
            self.encoder.conv1 = LoRAConv3d(
                self.encoder.conv1, rank=self.lora_rank, alpha=1.0
            )
            wrap_conv_with_lora(self.encoder.downsamples)
            wrap_attn_with_lora(self.encoder.downsamples)

        # "After" LoRA: bottleneck middle/head and full decoder (post-fusion path + decode).
        if self.use_lora_after:
            wrap_attn_with_lora(self.encoder.middle)
            wrap_conv_with_lora(self.encoder.middle)
            wrap_conv_with_lora(self.encoder.head)
            wrap_conv_with_lora(self.decoder)
            wrap_attn_with_lora(self.decoder)

    def encode(self, x, scale):
        """
        x: [B, V, 3, T, H, W]
        Returns: mu [B, z_dim, T', H', W']
        """
        b, v, c, t, h, w = x.shape
        assert v == self.num_views, f"AttentionMultiViewVideoVan expects {self.num_views} views, got {v}"

        # We do not use encoder.forward; manually unroll conv1 + downsamples.
        # Pass feat_cache so Wan's temporal stride-convs fire (temporal compression);
        # without feat_cache those convs are skipped and T is preserved (legacy).
        def run_down_path(x_in, feat_cache, feat_idx):
            if feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x_in[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                        dim=2,
                    )
                x_out = self.encoder.conv1(x_in, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x_out = self.encoder.conv1(x_in)
            for layer in self.encoder.downsamples:
                if feat_cache is not None:
                    x_out = layer(x_out, feat_cache, feat_idx)
                else:
                    x_out = layer(x_out)
            return x_out

        def encode_one_view(x_view):
            # x_view: [B, 3, T, H, W] -> bottleneck feats [B, Cb, T', H', W']
            if not self.temporal_compression:
                return run_down_path(x_view, None, [0])
            # Chunked time processing with a persistent per-view feat_cache:
            # frame 0 alone, then 4-frame chunks -> T' = 1 + (T-1)//4.
            feat_map = [None] * count_conv3d(self.encoder)
            t_in = x_view.shape[2]
            iter_ = 1 + (t_in - 1) // 4
            out = None
            for i in range(iter_):
                feat_idx = [0]
                if i == 0:
                    x_chunk = x_view[:, :, :1, :, :]
                else:
                    x_chunk = x_view[:, :, 1 + 4 * (i - 1): 1 + 4 * i, :, :]
                o = run_down_path(x_chunk, feat_map, feat_idx)
                out = o if out is None else torch.cat([out, o], dim=2)
            return out

        def encode_one_view_ckpt(x_view):
            if self.grad_checkpoint and self.training:
                return torch_checkpoint(encode_one_view, x_view, use_reentrant=False)
            return encode_one_view(x_view)

        with ProfileTimer.block("encode.downsample_all_views"):
            feats = [encode_one_view_ckpt(x[:, i]) for i in range(v)]

        # Keep time explicit for attention: [B, T', (H'*W'), C]
        def flatten_feat(feat):
            _, _, _, h2, w2 = feat.shape
            tokens = rearrange(feat, "b c t h w -> b t (h w) c")
            return tokens

        tokens_per_view = [flatten_feat(feat) for feat in feats]

        # Back to [B, C, T', H', W']
        def unflatten_tokens(tokens):
            b, t2, n, c = tokens.shape
            # Recover grid from feat_0 shape instead of hardcoding
            _, _, _, h2, w2 = feats[0].shape
            assert n == h2 * w2, "Token count mismatch when unflattening"
            return rearrange(tokens, "b t (h w) c -> b c t h w", h=h2, w=w2)

        if self.fusion_mode == "cross_attention":
            num_v = len(tokens_per_view)
            with ProfileTimer.block("encode.fusion.cross_attention"):
                # All views attend to each other first.
                stacked = torch.stack(tokens_per_view, dim=2)  # [B, T, V, N, C]
                enriched = self.cross_attn(stacked)  # [B, T, V, N, C]
                feats_enriched = [unflatten_tokens(enriched[:, :, v_idx]) for v_idx in range(num_v)]

            with ProfileTimer.block("encode.fusion.tree_merge"):
                current = feats_enriched
                merge_idx = 0  # global per-merge counter -> own (rb1, rb2) per merge/iteration
                while len(current) > 1:
                    next_level = []
                    for i in range(0, len(current) - 1, 2):
                        rb1, rb2 = self.tree_resblocks[merge_idx]
                        merged = torch.cat([current[i], current[i + 1]], dim=1)  # [B, 2C, T', H', W']
                        merged = rb1(merged)  # 2C → C
                        merged = rb2(merged)  # C  → C
                        next_level.append(merged)
                        merge_idx += 1
                    if len(current) % 2 == 1:
                        next_level.append(current[-1])  # carry odd view forward
                    current = next_level
                fused = current[0]
        elif self.fusion_mode == "self_attention":
            with ProfileTimer.block("encode.fusion.self_attention"):
                tokens_enriched = self.joint_attn(tokens_per_view)
                feats_enriched = [unflatten_tokens(tok) for tok in tokens_enriched]
                fused = torch.cat(feats_enriched, dim=1)
                fused = self.fusion_resblock1(fused)
                fused = self.fusion_resblock2(fused)
        elif self.fusion_mode == "conv3d":
            with ProfileTimer.block("encode.fusion.conv3d"):
                fused = torch.cat(feats, dim=1)  # [B,V*C,T',H',W']
                fused = self.view_conv_fuse(fused)
                b2, c2, t2, h2, w2 = fused.shape
                fused_2d = fused.permute(0, 2, 1, 3, 4).reshape(b2 * t2, c2, h2, w2)
                fused_2d = self.view_conv_norm(fused_2d)
                fused_2d = self.view_conv_act(fused_2d)
                fused = fused_2d.view(b2, t2, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
                fused = self.fusion_resblock1(fused)
                fused = self.fusion_resblock2(fused)
        else:
            with ProfileTimer.block("encode.fusion.conv4d"):
                x_4d = torch.stack(feats, dim=3)
                b2, c2, t2, v2, h2, w2 = x_4d.shape
                assert v2 == self.num_views, f"conv4d expected {self.num_views} views, got {v2}"

                # Spatial 2D conv: (B,V,T,C,H,W) -> (B*V*T,C,H,W)
                x_spatial = x_4d.permute(0, 3, 2, 1, 4, 5).reshape(b2 * v2 * t2, c2, h2, w2)
                x_spatial = self.conv4d_spatial(x_spatial)
                x_spatial = self.conv4d_norm(x_spatial)
                x_spatial = self.conv4d_act(x_spatial)
                x_spatial = x_spatial.view(b2, v2, t2, c2, h2, w2).permute(0, 1, 3, 2, 4, 5).contiguous()

                # Temporal 3D conv on (T,H,W): (B*V,C,T,H,W)
                x_temporal = x_spatial.reshape(b2 * v2, c2, t2, h2, w2)
                x_temporal = self.conv4d_temporal(x_temporal)
                x_temporal_2d = x_temporal.permute(0, 2, 1, 3, 4).reshape(b2 * v2 * t2, c2, h2, w2)
                x_temporal_2d = self.conv4d_norm(x_temporal_2d)
                x_temporal_2d = self.conv4d_act(x_temporal_2d)
                x_temporal = x_temporal_2d.view(b2 * v2, t2, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
                x_temporal = x_temporal.view(b2, v2, c2, t2, h2, w2)

                # View 3D conv on (V,H,W): (B*T,C,V,H,W), then squeeze V from 2 -> 1.
                x_view = x_temporal.permute(0, 3, 2, 1, 4, 5).reshape(b2 * t2, c2, v2, h2, w2)
                x_view = self.conv4d_view(x_view)
                x_view_2d = x_view.permute(0, 2, 1, 3, 4).reshape(b2 * t2, c2, h2, w2)
                x_view_2d = self.conv4d_norm(x_view_2d)
                x_view_2d = self.conv4d_act(x_view_2d)
                fused = x_view_2d.view(b2, t2, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()

                fused = self.fusion_resblock1(fused)
                fused = self.fusion_resblock2(fused)

        # Continue through the encoder middle and head (original code)
        with ProfileTimer.block("encode.middle"):
            x_mid = fused
            for layer in self.encoder.middle:
                x_mid = layer(x_mid)

        with ProfileTimer.block("encode.head"):
            x_head = x_mid
            for layer in self.encoder.head:
                x_head = layer(x_head)

        with ProfileTimer.block("encode.latent_conv"):
            out_conv = self.conv1(x_head)
            mu, log_var = out_conv.chunk(2, dim=1)

        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]

        if not self._encode_logged:
            self._encode_logged = True
            logger.info(
                "[AttentionMultiViewVideoVan] latent shape: %s  "
                "(B=%d, z_dim=%d, T'=%d, H'=%d, W'=%d)",
                tuple(mu.shape), mu.shape[0], mu.shape[1],
                mu.shape[2], mu.shape[3], mu.shape[4],
            )

        return mu, log_var

    def _encode_profiled(self, x, scale):
        """
        Same as encode() but wraps the full forward in torch.profiler.
        Call this manually on a warmed-up iteration to find bottlenecks.

        Prints a top-20 op table sorted by CUDA time and saves a Chrome
        trace to /tmp/encode_profile.json (open in chrome://tracing).
        """
        activities = [ProfilerActivity.CPU]
        if x.is_cuda:
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with record_function("encode_full"):
                mu, log_var = self.encode(x, scale)

        sort_key = "cuda_time_total" if x.is_cuda else "cpu_time_total"
        logger.info(
            "[AttentionMultiViewVideoVan] encode() profile — top 20 ops by %s:\n%s",
            sort_key,
            prof.key_averages().table(sort_by=sort_key, row_limit=20),
        )

        trace_path = "/tmp/encode_profile.json"
        prof.export_chrome_trace(trace_path)
        logger.info("[AttentionMultiViewVideoVan] Chrome trace saved to %s", trace_path)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, scale, view_idx: int = 0):
        """
        z: [B, z_dim, T', H', W']
        Returns: reconstruction [B, 3, T, H, W]
        """
        if not (0 <= view_idx < self.num_views):
            raise IndexError(f"view_idx={view_idx} out of range for num_views={self.num_views}")
        if self.use_viewwise_decoder_lora:
            with ProfileTimer.block("decode.view_condition"):
                lora_delta = self.view_lora_up[view_idx](self.view_lora_down[view_idx](z))
                z = z + lora_delta

        # VIEW_EMBEDDINGS 2: Lookup + add + decode
        else:
            with ProfileTimer.block("decode.view_condition"):
                view_idx_tensor = torch.tensor(view_idx, device=z.device, dtype=torch.long)
                emb = self.view_embed(view_idx_tensor).view(1, self.z_dim, 1, 1, 1)
                z = z + emb

        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]

        def decode_body(z_cond):
            # conv2 on the full latent, then per-latent-frame decode. With
            # temporal_compression a persistent feat_cache makes the temporal
            # upsample convs fire (T' -> ~4*T'); otherwise 1 latent frame -> 1 out frame.
            x = self.conv2(z_cond)
            iter_ = x.shape[2]
            feat_map = [None] * count_conv3d(self.decoder) if self.temporal_compression else None
            out = None
            for i in range(iter_):
                x_chunk = x[:, :, i : i + 1, :, :]
                if feat_map is not None:
                    conv_idx = [0]
                    o = self.decoder(x_chunk, feat_cache=feat_map, feat_idx=conv_idx)
                else:
                    o = self.decoder(x_chunk)
                out = o if out is None else torch.cat([out, o], 2)
            return out

        with ProfileTimer.block("decode.temporal_loop"):
            if self.grad_checkpoint and self.training:
                out = torch_checkpoint(decode_body, z, use_reentrant=False)
            else:
                out = decode_body(z)
        return out


    def forward(self, x, scale):
        """
        Full VAE forward.

        Args:
            x: [B, 2, 3, T, H, W]
            scale: (mean, inv_std) as in WanVideoVAE.
        """
        mu, log_var = self.encode(x, scale)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, scale)
        return x_recon, mu, log_var


class WanVideoVAE(nn.Module):

    def __init__(self, z_dim=16):
        super().__init__()

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = VideoVAE_(z_dim=z_dim).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.z_dim = z_dim


    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x


    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask


    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) * self.upsampling_factor, (size_w - stride_w) * self.upsampling_factor)
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values


    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)
        values = torch.zeros((1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) // self.upsampling_factor, (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values


    def single_encode(self, video, device):
        video = video.to(device)
        x = self.model.encode(video, self.scale)
        return x


    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)


    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
                tile_stride = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states


    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            video = video.squeeze(0)
            videos.append(video)
        videos = torch.stack(videos)
        return videos


    @staticmethod
    def state_dict_converter():
        return WanVideoVAEStateDictConverter()

class MultiViewWanVideoVAE(nn.Module):
    """
    Multi-view Wan VAE with view-aware latent compression.

    This mirrors WanVideoVAE's public interface but expects videos shaped
    [B, V, C, T, H, W]. Tiling is intentionally not supported yet because
    stitching across both spatial and view dimensions needs extra bookkeeping.
    """

    def __init__(self, z_dim=16, view_in=8, view_compression=4, use_view_embedding=True):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # Convert a compression factor into an integer view count (e.g. 2->1).
        view_out = max(1, int(view_in) // max(1, int(view_compression)))
        self.model = MultiViewVideoVAE_(
            z_dim=z_dim,
            view_in=view_in,
            view_out=view_out,
            use_view_embedding=use_view_embedding,
        ).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.z_dim = z_dim
        self.view_in = int(view_in)
        self.view_out = int(view_out)

    def single_encode(self, video, device):
        video = video.to(device)
        return self.model.encode(video, self.scale)

    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        return self.model.decode(hidden_state, self.scale).clamp_(-1, 1)

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        if tiled:
            raise NotImplementedError(
            # Allow single-view inputs by inserting a view dimension.
                "Multi-view tiling is not supported yet; use tiled=False for now."
            )
        if isinstance(videos, (list, tuple)):
            videos = torch.stack(videos)
        if videos.dim() == 5:
            videos = videos.unsqueeze(1)
        return self.single_encode(videos, device)

    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        if tiled:
            raise NotImplementedError(
                "Multi-view tiling is not supported yet; use tiled=False for now."
            )
        if isinstance(hidden_states, (list, tuple)):
            hidden_states = torch.stack(hidden_states)
        return self.single_decode(hidden_states, device)
    
    @staticmethod
    def state_dict_converter():
        return WanVideoVAEStateDictConverter()

# class MultiViewWanVideoVAE(nn.Module):

#     def __init__(
#         self,
#         base_vae,          # pass a normal WanVideoVAE
#         view_in=2,
#         freeze_temporal=True,
#     ):
#         super().__init__()

#         self.base = base_vae
#         self.view_in = view_in
#         self.z_dim = base_vae.z_dim

#         # 🔥 Latent fusion (V → 1)
#         self.latent_fusion = nn.Conv3d(
#             in_channels=view_in * self.z_dim,
#             out_channels=self.z_dim,
#             kernel_size=1,
#             bias=False,
#         )

#         # 🔥 Latent expansion (1 → V)
#         self.latent_expand = nn.Conv3d(
#             in_channels=self.z_dim,
#             out_channels=view_in * self.z_dim,
#             kernel_size=1,
#             bias=False,
#         )

#         # Freeze only temporal modules
#         if freeze_temporal:
#             for name, param in self.base.named_parameters():
#                 if "temporal" in name.lower():
#                     param.requires_grad = False

#     # -----------------------------------
#     # ENCODE
#     # -----------------------------------
#     def encode(self, x):

#         # x: [B, V, C, T, H, W]
#         B, V, C, T, H, W = x.shape

#         latents = []
#         for v in range(V):
#             z_v = self.base.encode(x[:, v])
#             latents.append(z_v)

#         # [B, V, Z, T', H', W']
#         z_stack = torch.stack(latents, dim=1)

#         B, V, Z, T2, H2, W2 = z_stack.shape

#         # channel-fuse
#         z_stack = z_stack.view(B, V * Z, T2, H2, W2)
#         z = self.latent_fusion(z_stack)

#         return z

#     # -----------------------------------
#     # DECODE
#     # -----------------------------------
#     def decode(self, z):

#         # expand
#         z_expand = self.latent_expand(z)

#         B, _, T2, H2, W2 = z_expand.shape
#         Z = self.z_dim

#         z_expand = z_expand.view(B, self.view_in, Z, T2, H2, W2)

#         recons = []
#         for v in range(self.view_in):
#             x_v = self.base.decode(z_expand[:, v])
#             recons.append(x_v)

#         return torch.stack(recons, dim=1)

#     def forward(self, x):
#         z = self.encode(x)
#         x_rec = self.decode(z)
#         return x_rec
        

class WanVideoVAEStateDictConverter:

    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        state_dict_ = {}
        if 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        for name in state_dict:
            state_dict_['model.' + name] = state_dict[name]
        return state_dict_


class VideoVAE38_(VideoVAE_):

    def __init__(self,
                 dim=160,
                 z_dim=48,
                 dec_dim=256,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[False, True, True],
                 dropout=0.0):
        super(VideoVAE_, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d_38(dim, z_dim * 2, dim_mult, num_res_blocks,
                                    attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d_38(dec_dim, z_dim, dim_mult, num_res_blocks,
                                    attn_scales, self.temperal_upsample, dropout)


    def encode(self, x, scale):
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :],
                                   feat_cache=self._enc_feat_map,
                                   feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                    feat_cache=self._enc_feat_map,
                                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu


    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :],
                                   feat_cache=self._feat_map,
                                   feat_idx=self._conv_idx,
                                   first_chunk=True)
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :],
                                    feat_cache=self._feat_map,
                                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out


class WanVideoVAE38(WanVideoVAE):

    def __init__(self, z_dim=48, dim=160):
        super(WanVideoVAE, self).__init__()

        mean = [
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
            -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
            -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
            0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667
        ]
        std = [
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = VideoVAE38_(z_dim=z_dim, dim=dim).eval().requires_grad_(False)
        self.upsampling_factor = 16
        self.z_dim = z_dim