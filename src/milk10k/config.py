"""Single source of truth for paths, seed and image size.

Homework rule: *never hard-code absolute paths*. Every path in the project is
derived from :data:`PROJECT_ROOT` (the repo root, found relative to this file)
or from the ``MILK10K_DATA`` environment variable, which lets the data live
outside the repo without touching a line of code::

    export MILK10K_DATA=/Volumes/ssd/milk10k     # optional
    python -m scripts.build_pipeline

Nothing else in the project may call ``os.environ`` or build a path literal:
if you need a new location, add it here.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

#: Repo root = two levels up from src/milk10k/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Dataset root. Override with MILK10K_DATA; defaults to the in-repo (git-ignored)
#: copy so a fresh clone + download works with no configuration at all.
DATA_DIR: Path = Path(
    os.environ.get("MILK10K_DATA", PROJECT_ROOT / "data" / "raw" / "milk10k")
).expanduser()

IMAGE_DIR: Path = DATA_DIR / "images"
METADATA_CSV: Path = DATA_DIR / "metadata.csv"
SUPPLEMENT_DIR: Path = DATA_DIR / "supplements"

#: Generated outputs. Kept apart from source code on purpose (B0):
#: ``artifacts/`` is committed (small, it is the evidence), ``reports/figures``
#: holds the PNGs, ``data/processed`` holds bulky intermediates (git-ignored).
ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"
SPLIT_DIR: Path = ARTIFACT_DIR / "splits"
FIG_DIR: Path = PROJECT_ROOT / "reports" / "figures"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
REPORT_DIR: Path = PROJECT_ROOT / "reports"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

#: The one seed. Splits, samplers and any shuffling derive from it.
SEED: int = 42

#: Date the committed splits were generated (written into the README, B5).
SPLIT_DATE: str = "2026-09-24"


def set_seed(seed: int = SEED) -> int:
    """Seed ``random``, ``numpy`` and (if installed) ``torch``. Returns the seed."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the pandas-only notebooks
        pass
    return seed


# --------------------------------------------------------------------------
# Pipeline hyper-parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Everything a later milestone needs to rebuild the exact same pipeline.

    Defaults are justified in the README and the Milestone 1 report:
    ``image_size`` 224 because every MILK10k image is 600x450, so 224 is a pure
    downscale for 100% of the set (B6); the split sizes give a test set big
    enough that the rarest class (MAL_OTH, 9 lesions) is still representable.
    """

    image_size: int = 224
    seed: int = SEED
    #: 1/7 rather than the conventional 0.15. A grouped, stratified split can
    #: only hold out whole folds, and the rarest class (MAL_OTH, 9 lesions)
    #: caps the fold count at 9 - so the achievable fractions are quantised to
    #: 1/n. 1/7 = 0.1429 is the achievable value nearest 15%, and using it makes
    #: the realised split sizes exact instead of ~1.4pp off. See splits.py.
    val_size: float = 1 / 7
    test_size: float = 1 / 7
    batch_size: int = 32
    num_workers: int = 0  # 0 = deterministic + safe inside notebooks
    #: Primary target. "diagnosis_1" = Benign/Malignant/Indeterminate.
    target: str = "diagnosis_1"
    #: Label used for *stratifying* the split: the 11-class scheme is finer,
    #: so stratifying on it automatically balances diagnosis_1 as well (B5).
    stratify_on: str = "dx"
    #: ImageNet statistics; replaced by train-only stats at build time (B8).
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if not 0 < self.val_size < 1 or not 0 < self.test_size < 1:
            raise ValueError("val_size and test_size must be fractions in (0, 1)")
        if self.val_size + self.test_size >= 1:
            raise ValueError("val_size + test_size must leave room for train")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")


#: The configuration the committed artefacts were built with.
CONFIG = PipelineConfig()


def ensure_dirs() -> None:
    """Create every generated-output directory (idempotent)."""
    for d in (ARTIFACT_DIR, SPLIT_DIR, FIG_DIR, PROCESSED_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable config dump - printed at the top of the notebooks."""
    lines = [
        f"project root : {PROJECT_ROOT}",
        f"data dir     : {DATA_DIR}"
        + ("  (from $MILK10K_DATA)" if "MILK10K_DATA" in os.environ else "  (default)"),
        f"images       : {IMAGE_DIR}",
        f"artifacts    : {ARTIFACT_DIR}",
        f"figures      : {FIG_DIR}",
        f"seed         : {SEED}",
        f"image size   : {CONFIG.image_size}",
    ]
    return "\n".join(lines)
