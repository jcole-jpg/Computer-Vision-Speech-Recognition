"""MILK10k project package.

    config      paths (MILK10K_DATA), seed, image size - the ONE place for settings
    data        paths, class map, metadata loaders, merged tables, available_subset
    stats       metadata <-> target association tests            (homework Part 1)
    color       dataset-level histogram / colour statistics      (homework Part 2)
    preprocess  single-image + batch preprocessing pipeline      (homework Part 3)
    loader      MILK10kLoader: batches of (image, label)         (homework Part 4)
    viz         image grids, class balance, batch summaries      (homework Part 5)
    integrity   label cross-checks + full-image-set verification  (A1.1, B1)
    imaging     NumPy geometry, PSNR, JPEG quality, file sizes    (A2)
    splits      grouped/stratified lesion-level splitting         (A3.2-A3.3, B5)
    leakage     what a lesion leak does to a real metric          (A3.1)
    metrics     dummy baselines and cost-aware evaluation         (A3.4)
    augment     train/eval transforms + augmentation audit        (A3.5, B7)
    labels      label strategy, label map, class weights          (B3)
    datasets    LesionDataset, image Dataset, DataLoaders         (A3.6, B8)

Everything from ``data`` is re-exported here so the Session 1-2 notebooks'
``import milk10k`` / ``from src import milk10k`` keep working unchanged.

``augment`` and ``datasets`` are deliberately NOT auto-imported: they pull in
torch, and the Session 1-2 notebooks are pandas-only. Import them explicitly::

    from milk10k import augment, datasets
"""

from .data import *  # noqa: F401,F403  (constants + loaders)
from .data import (  # explicit names for IDEs
    PROJECT_ROOT, DATA_DIR, IMAGE_DIR, FIG_DIR,
    CLASS_NAMES, CLASS_CODES, MALIGNANT, SKIN_TONE_LABELS,
    BLUE, ORANGE, AQUA, BENIGN_COLOR, MALIGNANT_COLOR, SEQ_CMAP,
    load_metadata, load_training_input, load_training_gt, load_training_supp,
    lesion_labels, load_images_table, load_lesions_table, available_subset,
)
from .config import CONFIG, PipelineConfig, SEED, ARTIFACT_DIR, SPLIT_DIR, ensure_dirs, set_seed
from .data import build_lesion_table, assert_lesion_table
from .preprocess import PreprocessConfig, ImageInfo, BatchResult, preprocess_image, preprocess_batch
from .loader import MILK10kLoader, Batch
from . import (  # noqa: E402  (submodules, so milk10k.viz.* works)
    stats, color, viz, integrity, imaging, splits, leakage, metrics, labels,
)

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "IMAGE_DIR", "FIG_DIR",
    "CLASS_NAMES", "CLASS_CODES", "MALIGNANT", "SKIN_TONE_LABELS",
    "BLUE", "ORANGE", "AQUA", "BENIGN_COLOR", "MALIGNANT_COLOR", "SEQ_CMAP",
    "load_metadata", "load_training_input", "load_training_gt", "load_training_supp",
    "lesion_labels", "load_images_table", "load_lesions_table", "available_subset",
    "PreprocessConfig", "ImageInfo", "BatchResult", "preprocess_image", "preprocess_batch",
    "MILK10kLoader", "Batch",
    "CONFIG", "PipelineConfig", "SEED", "ARTIFACT_DIR", "SPLIT_DIR",
    "ensure_dirs", "set_seed", "build_lesion_table", "assert_lesion_table",
]
