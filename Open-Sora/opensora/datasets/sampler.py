"""Reconstructed minimal sampler module (the upstream file is gitignored and
was absent from this checkout; the cluster working tree had the full package).

The VAE trainer's PtVideoDataset path in ``prepare_dataloader`` only needs a
plain ``DistributedSampler`` (``set_step``/``reset`` uses in train.py are
hasattr-guarded). ``VariableVideoBatchSampler`` is only instantiated for the
bucketized ``VideoTextDataset`` path, which this fork never takes.
"""
from torch.utils.data.distributed import DistributedSampler


class VariableVideoBatchSampler:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VariableVideoBatchSampler was stripped from this fork; only the "
            "PtVideoDataset (fixed-size) dataloader path is supported."
        )


__all__ = ["DistributedSampler", "VariableVideoBatchSampler"]
