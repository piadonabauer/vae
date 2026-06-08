import torch
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from torch.optim.lr_scheduler import _LRScheduler

# Try to import HybridAdam, but fall back to standard optimizers if it fails
try:
    from colossalai.nn.optimizer import HybridAdam
    HYBRID_ADAM_AVAILABLE = True
except (ImportError, AssertionError) as e:
    HYBRID_ADAM_AVAILABLE = False
    print(f"Warning: HybridAdam not available ({e}). Falling back to standard PyTorch optimizers.")


def create_optimizer(
    model: torch.nn.Module,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """
    Create an optimizer.

    Args:
        model (torch.nn.Module): The model to be optimized.
        optimizer_config (dict): The configuration of the optimizer.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    optimizer_name = optimizer_config.pop("cls", "HybridAdam")
    config_copy = optimizer_config.copy()
    
    # Handle HybridAdam (requires CUDA_HOME)
    if optimizer_name == "HybridAdam":
        if HYBRID_ADAM_AVAILABLE:
            try:
                optimizer_cls = HybridAdam
                optimizer = optimizer_cls(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    **config_copy,
                )
                return optimizer
            except (AssertionError, RuntimeError) as e:
                if "CUDA_HOME" in str(e) or "CUDA" in str(e):
                    print(f"Warning: HybridAdam requires CUDA_HOME which is not set.")
                    print(f"Falling back to AdamW. To use HybridAdam, set CUDA_HOME environment variable.")
                    print(f"On Compute Canada, try: export CUDA_HOME=$CUDA_HOME or module load cuda")
                    optimizer_name = "AdamW"  # Fall through to standard optimizer
                else:
                    raise
    
    # Standard PyTorch optimizers (don't require CUDA extensions)
    if optimizer_name == "AdamW" or optimizer_name == "Adam":
        # Map ColossalAI config to PyTorch config
        # Remove ColossalAI-specific params and use standard ones
        torch_config = {}
        if "lr" in config_copy:
            torch_config["lr"] = config_copy.pop("lr")
        if "eps" in config_copy:
            torch_config["eps"] = config_copy.pop("eps")
        if "weight_decay" in config_copy:
            torch_config["weight_decay"] = config_copy.pop("weight_decay")
        if "betas" in config_copy:
            torch_config["betas"] = config_copy.pop("betas")
        # Remove adamw_mode - PyTorch AdamW is always AdamW mode
        config_copy.pop("adamw_mode", None)
        
        # Warn about unused config
        if config_copy:
            print(f"Warning: Unused optimizer config keys: {list(config_copy.keys())}")
        
        optimizer_cls = torch.optim.AdamW if optimizer_name == "AdamW" else torch.optim.Adam
        optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, model.parameters()),
            **torch_config,
        )
        return optimizer
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: HybridAdam, AdamW, Adam")


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps_per_epoch: int,
    epochs: int = 1000,
    warmup_steps: int | None = None,
    use_cosine_scheduler: bool = False,
    use_exponential_decay: bool = False,
    initial_lr: float = 1e-6,
    min_lr: float = 0.0,
    total_steps: int | None = None,
    decay_steps: int = 5000,
    decay_factor: float = 0.5,
) -> _LRScheduler | None:
    """
    Create a learning rate scheduler.

    Three modes (mutually exclusive, checked in order):
      1. use_exponential_decay=True  — halves every decay_steps optimizer steps, independent of
                                       total training length. Best when you don't know how long
                                       you'll train. Decays to min_lr then stays there.
      2. use_cosine_scheduler=True   — cosine decay from peak to min_lr over total_steps.
                                       Requires knowing (or estimating) training length.
      3. warmup_steps only           — linear warmup then constant LR.

    Args:
        optimizer: The optimizer to be used.
        num_steps_per_epoch: Steps per epoch (only used by cosine when total_steps is None).
        epochs: Total epochs (only used by cosine when total_steps is None).
        warmup_steps: Short linear warmup before the main schedule. 0 or None = skip.
        use_exponential_decay: Smooth per-step exponential decay. After warmup, every
            decay_steps steps the LR is multiplied by decay_factor (e.g. 0.5 = halve).
        use_cosine_scheduler: Cosine annealing decay over total_steps.
        initial_lr: LR at the start of the warmup ramp.
        min_lr: Floor LR — decay stops here.
        total_steps: Explicit cycle length for cosine. Falls back to num_steps_per_epoch * epochs.
        decay_steps: For exponential decay: interval (optimizer steps) between each halving.
        decay_factor: For exponential decay: multiplicative factor per decay_steps interval.
    """
    if not warmup_steps and not use_cosine_scheduler and not use_exponential_decay:
        return None

    if use_exponential_decay:
        return ExponentialDecayLR(
            optimizer,
            warmup_steps=warmup_steps or 0,
            decay_steps=decay_steps,
            decay_factor=decay_factor,
            min_lr=min_lr,
            initial_lr=initial_lr,
        )

    if use_cosine_scheduler:
        effective_total_steps = total_steps if total_steps is not None else num_steps_per_epoch * epochs
        return CosineAnnealingWarmupLR(
            optimizer,
            total_steps=effective_total_steps,
            warmup_steps=warmup_steps or 0,
            eta_min=min_lr,
        )

    return LinearWarmupLR(optimizer, initial_lr=initial_lr, warmup_steps=warmup_steps)


class LinearWarmupLR(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_step (int, optional): The index of last step, defaults to -1. When last_step=-1,
            the schedule is started from the beginning or When last_step=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, initial_lr=0, warmup_steps: int = 0, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr + (self.last_epoch + 1) / (self.warmup_steps + 1) * (lr - self.initial_lr)
                for lr in self.base_lrs
            ]
        else:
            return self.base_lrs


class ExponentialDecayLR(_LRScheduler):
    """Smooth exponential LR decay that needs no knowledge of total training length.

    After an optional linear warmup, the LR decays continuously so that every
    ``decay_steps`` optimizer steps it is multiplied by ``decay_factor`` (e.g. 0.5 = halve).
    Decay stops at ``min_lr``.

    Example with decay_steps=5000, decay_factor=0.5, peak_lr=5e-4, min_lr=2.5e-5:
        step     0  → warmup
        step   100  → 5e-4   (peak)
        step  5100  → 2.5e-4 (×0.5)
        step 10100  → 1.25e-4
        step 15100  → 6.25e-5
        step 20100  → 2.5e-5  → stays here
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 0,
        decay_steps: int = 5000,
        decay_factor: float = 0.5,
        min_lr: float = 0.0,
        initial_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        # Per-step gamma so that after decay_steps steps LR = base_lr * decay_factor.
        self.gamma = decay_factor ** (1.0 / max(1, decay_steps))
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            progress = (self.last_epoch + 1) / (self.warmup_steps + 1)
            return [self.initial_lr + progress * (base - self.initial_lr) for base in self.base_lrs]
        steps_after_warmup = self.last_epoch - self.warmup_steps
        return [max(self.min_lr, base * (self.gamma ** steps_after_warmup)) for base in self.base_lrs]