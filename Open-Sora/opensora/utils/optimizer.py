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
    initial_lr: float = 1e-6,
) -> _LRScheduler | None:
    """
    Create a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        num_steps_per_epoch (int): The number of steps per epoch.
        epochs (int): The number of epochs.
        warmup_steps (int |  None): The number of warmup steps.
        use_cosine_scheduler (bool): Whether to use cosine scheduler.

    Returns:
        _LRScheduler |  None: The learning rate scheduler
    """
    if warmup_steps is None and not use_cosine_scheduler:
        lr_scheduler = None
    elif use_cosine_scheduler:
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer,
            total_steps=num_steps_per_epoch * epochs,
            warmup_steps=warmup_steps,
        )
    else:
        lr_scheduler = LinearWarmupLR(optimizer, initial_lr=1e-6, warmup_steps=warmup_steps)
        # lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)

    return lr_scheduler


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