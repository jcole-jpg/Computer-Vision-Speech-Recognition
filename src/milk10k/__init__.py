"""MILK10k project package.

    data        paths, class map, metadata loaders, merged tables, available_subset
    stats       metadata <-> target association tests            (homework Part 1)
    color       dataset-level histogram / colour statistics      (homework Part 2)
    preprocess  single-image + batch preprocessing pipeline      (homework Part 3)
    loader      MILK10kLoader: batches of (image, label)         (homework Part 4)
    viz         image grids, class balance, batch summaries      (homework Part 5)

Everything from ``data`` is re-exported here so the Session 1-2 notebooks'
``import milk10k`` / ``from src import milk10k`` keep working unchanged.
"""

from .data import *  # noqa: F401,F403  (constants + loaders)
from .data import (  # explicit names for IDEs
    PROJECT_ROOT, DATA_DIR, IMAGE_DIR, FIG_DIR,
    CLASS_NAMES, CLASS_CODES, MALIGNANT, SKIN_TONE_LABELS,
    BLUE, ORANGE, AQUA, BENIGN_COLOR, MALIGNANT_COLOR, SEQ_CMAP,
    load_metadata, load_training_input, load_training_gt, load_training_supp,
    lesion_labels, load_images_table, load_lesions_table, available_subset,
)
from .preprocess import PreprocessConfig, ImageInfo, BatchResult, preprocess_image, preprocess_batch
from .loader import MILK10kLoader, Batch
from . import stats, color, viz  # noqa: E402  (submodules, so milk10k.viz.* works)

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "IMAGE_DIR", "FIG_DIR",
    "CLASS_NAMES", "CLASS_CODES", "MALIGNANT", "SKIN_TONE_LABELS",
    "BLUE", "ORANGE", "AQUA", "BENIGN_COLOR", "MALIGNANT_COLOR", "SEQ_CMAP",
    "load_metadata", "load_training_input", "load_training_gt", "load_training_supp",
    "lesion_labels", "load_images_table", "load_lesions_table", "available_subset",
    "PreprocessConfig", "ImageInfo", "BatchResult", "preprocess_image", "preprocess_batch",
    "MILK10kLoader", "Batch",
]
