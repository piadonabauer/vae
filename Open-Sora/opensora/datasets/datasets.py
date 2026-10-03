"""Reconstructed minimal datasets module (the upstream file is gitignored and
was absent from this checkout; the cluster working tree had the full package).

``TextDataset`` / ``VideoTextDataset`` are referenced only in isinstance
checks inside ``dataloader.prepare_dataloader``; this fork always feeds
``PtVideoDataset``, so plain placeholder classes are sufficient and the
isinstance checks simply evaluate to False.
"""


class TextDataset:
    pass


class VideoTextDataset:
    pass
