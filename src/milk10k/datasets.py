"""PyTorch datasets and loaders (A3.6, B8).

Two datasets, for two units of analysis:

* :class:`LesionDataset` - one item per **lesion**, returning both views. This is
  the correct unit for this project: the label was assigned to the lesion, and a
  clinician decides per lesion, not per photograph.
* :class:`MILK10kImageDataset` - one item per **image**, the upgrade of the
  Session 2 loader. Convenient for training (twice the samples, and each view is
  informative on its own); its predictions are aggregated back to lesion level
  with :func:`aggregate_predictions` before anything is reported.

Both fail loudly on a missing file. A skipped image silently shrinks the dataset
and quietly changes the class balance, which is far worse than a crash.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augment import eval_transform, train_transform
from .config import CONFIG, IMAGE_DIR, SEED


class MissingImageError(FileNotFoundError):
    """Raised when a table row points at an image that is not on disk."""


def _load_rgb(path: Path, isic_id: str) -> Image.Image:
    path = Path(path)
    if not path.is_file():
        raise MissingImageError(
            f"image for {isic_id} not found at {path}. "
            "The pipeline fails loudly on purpose: silently skipping a row would "
            "shrink the dataset and shift the class balance without warning. "
            "Pass allow_missing=True only for a deliberate partial-data run."
        )
    with Image.open(path) as im:
        return im.convert("RGB")


# --------------------------------------------------------------------------
# A3.6 - lesion-level dataset
# --------------------------------------------------------------------------


class LesionDataset(Dataset):
    """One item per lesion, with one tensor per requested view (A3.6).

    Each item is a dict::

        {"derm": Tensor(3,H,W), "clinical": Tensor(3,H,W),
         "label": int, "lesion_id": str}

    The transform is applied **independently** to each view, so the two views of
    a lesion get different random augmentations - they are different photographs
    and should be perturbed as such.
    """

    #: view name -> column in the lesion table holding that view's isic_id
    VIEW_COLUMNS = {"derm": "derm_id", "clinical": "clinical_id"}

    def __init__(self, lesion_table: pd.DataFrame, img_dir: Path | str = IMAGE_DIR,
                 label_col: str = "diagnosis_1", transform=None,
                 views: tuple[str, ...] = ("derm", "clinical"),
                 label_map: dict[str, int] | None = None):
        unknown = set(views) - set(self.VIEW_COLUMNS)
        if unknown:
            raise ValueError(f"unknown views {sorted(unknown)}; "
                             f"expected any of {sorted(self.VIEW_COLUMNS)}")
        missing_cols = [self.VIEW_COLUMNS[v] for v in views
                        if self.VIEW_COLUMNS[v] not in lesion_table.columns]
        if missing_cols:
            raise KeyError(f"lesion table is missing columns {missing_cols}")
        if label_col not in lesion_table.columns:
            raise KeyError(f"lesion table has no label column {label_col!r}")

        self.table = lesion_table.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.label_col = label_col
        self.transform = transform if transform is not None else eval_transform()
        self.views = tuple(views)

        classes = sorted(self.table[label_col].dropna().unique())
        self.label_map = label_map if label_map is not None else {c: i for i, c in enumerate(classes)}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict:
        row = self.table.iloc[idx]
        item: dict = {
            "label": int(self.label_map[row[self.label_col]]),
            "lesion_id": str(row["lesion_id"]),
        }
        for view in self.views:
            isic_id = row[self.VIEW_COLUMNS[view]]
            img = _load_rgb(self.img_dir / f"{isic_id}.jpg", isic_id)
            item[view] = self.transform(img)  # applied independently per view
        return item

    def class_counts(self) -> pd.Series:
        return self.table[self.label_col].value_counts()


def aggregate_predictions(image_probs: np.ndarray, image_table: pd.DataFrame,
                          lesion_col: str = "lesion_id",
                          classes: list[str] | None = None) -> pd.DataFrame:
    """Turn per-image class probabilities into ONE prediction per lesion (A3.6c).

    The two views are averaged: each is an independent look at the same lesion,
    and averaging probabilities is the standard way to pool them without letting
    one confident-but-wrong view dominate a vote.

    ``image_probs`` is (n_images, n_classes) and aligned row-for-row with
    ``image_table``. Returns one row per lesion with the mean probabilities, the
    predicted class and the number of views that contributed.
    """
    probs = np.asarray(image_probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"image_probs must be 2-D (n_images, n_classes), got {probs.shape}")
    if len(probs) != len(image_table):
        raise ValueError(f"image_probs has {len(probs)} rows but image_table has "
                         f"{len(image_table)}")

    n_classes = probs.shape[1]
    cols = classes if classes is not None else [f"p{i}" for i in range(n_classes)]
    if len(cols) != n_classes:
        raise ValueError(f"{len(cols)} class names for {n_classes} probability columns")

    df = pd.DataFrame(probs, columns=cols)
    df[lesion_col] = image_table[lesion_col].to_numpy()

    out = df.groupby(lesion_col, sort=True)[cols].mean()
    out["n_views"] = df.groupby(lesion_col, sort=True).size()
    out["pred"] = out[cols].to_numpy().argmax(axis=1)
    out["pred_label"] = [cols[i] for i in out["pred"]]
    return out.reset_index()


# --------------------------------------------------------------------------
# B8 - image-level dataset + the three dataloaders
# --------------------------------------------------------------------------


class MILK10kImageDataset(Dataset):
    """One item per image: ``(image_tensor, label:int, isic_id:str)`` (B8).

    The Session 2 loader did manual resize/normalise arithmetic and skipped rows
    whose file was absent. This upgrade keeps its shape but takes transforms from
    torchvision, reads integer labels through the committed label map, returns
    the ``isic_id`` so a wrong prediction can be traced back to a file, and
    raises on a missing image instead of quietly dropping it.
    """

    def __init__(self, split_df: pd.DataFrame, img_dir: Path | str = IMAGE_DIR,
                 label_col: str = "diagnosis_1",
                 label_map: dict[str, int] | None = None,
                 transform=None, allow_missing: bool = False):
        if "isic_id" not in split_df.columns:
            raise KeyError("split table must have an isic_id column")
        if label_col not in split_df.columns:
            raise KeyError(f"split table has no label column {label_col!r}")

        self.img_dir = Path(img_dir)
        self.label_col = label_col
        self.transform = transform if transform is not None else eval_transform()
        self.allow_missing = allow_missing

        table = split_df.reset_index(drop=True)
        if allow_missing:
            exists = table["isic_id"].map(lambda i: (self.img_dir / f"{i}.jpg").is_file())
            table = table[exists].reset_index(drop=True)
        self.table = table

        classes = sorted(self.table[label_col].dropna().unique())
        self.label_map = label_map if label_map is not None else {c: i for i, c in enumerate(classes)}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.table.iloc[idx]
        isic_id = str(row["isic_id"])
        img = _load_rgb(self.img_dir / f"{isic_id}.jpg", isic_id)
        return self.transform(img), int(self.label_map[row[self.label_col]]), isic_id

    @property
    def labels(self) -> np.ndarray:
        return self.table[self.label_col].map(self.label_map).to_numpy()

    def class_counts(self) -> pd.Series:
        return self.table[self.label_col].value_counts()


def make_weighted_sampler(dataset: MILK10kImageDataset) -> WeightedRandomSampler:
    """Sample rare classes more often, so a batch is roughly balanced (B8).

    Per-sample weight = 1 / count(its class). Drawn with replacement, so a
    9-lesion class reappears many times per epoch instead of being seen twice.
    Use this **or** weighted loss, not usually both: each already corrects the
    imbalance once, and stacking them over-corrects toward the rare classes.
    """
    y = dataset.labels
    counts = np.bincount(y, minlength=len(dataset.label_map)).astype(np.float64)
    per_class = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    weights = per_class[y]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(weights), replacement=True)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic, distinct seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    datasets: dict[str, MILK10kImageDataset]


def build_dataloaders(split_tables: dict[str, pd.DataFrame],
                      label_map: dict[str, int],
                      img_dir: Path | str = IMAGE_DIR,
                      label_col: str = "diagnosis_1",
                      image_size: int = CONFIG.image_size,
                      batch_size: int = CONFIG.batch_size,
                      num_workers: int = CONFIG.num_workers,
                      seed: int = SEED,
                      use_sampler: bool = True,
                      norm: tuple[tuple[float, ...], tuple[float, ...]] | None = None
                      ) -> Loaders:
    """Three DataLoaders: train (shuffled/augmented), val and test (deterministic).

    Augmentation is applied to train only - augmenting evaluation data would make
    the metric depend on a random draw and stop it being comparable between runs.
    """
    mean, std = norm if norm is not None else (CONFIG.norm_mean, CONFIG.norm_std)
    tf_train = train_transform(image_size, mean, std)
    tf_eval = eval_transform(image_size, mean, std)

    ds = {
        name: MILK10kImageDataset(
            df, img_dir=img_dir, label_col=label_col, label_map=label_map,
            transform=tf_train if name == "train" else tf_eval,
        )
        for name, df in split_tables.items()
    }

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = make_weighted_sampler(ds["train"]) if use_sampler else None
    train_loader = DataLoader(
        ds["train"], batch_size=batch_size,
        shuffle=(sampler is None),  # a sampler already defines the order
        sampler=sampler, num_workers=num_workers, drop_last=False,
        worker_init_fn=seed_worker, generator=generator,
    )
    eval_loaders = {
        name: DataLoader(ds[name], batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, worker_init_fn=seed_worker)
        for name in ("val", "test") if name in ds
    }
    return Loaders(train_loader, eval_loaders["val"], eval_loaders["test"], ds)


def sanity_report(loaders: Loaders, n_label_batches: int = 20,
                  time_one_epoch: bool = True) -> dict:
    """The printed sanity checks B8 asks for: shapes, dtype, range, balance, timing."""
    images, labels, ids = next(iter(loaders.train))
    inv = loaders.datasets["train"].inverse_label_map

    hist: dict[str, int] = {}
    for i, (_, y, _) in enumerate(loaders.train):
        if i >= n_label_batches:
            break
        for c, n in zip(*np.unique(y.numpy(), return_counts=True)):
            hist[inv[int(c)]] = hist.get(inv[int(c)], 0) + int(n)

    report = {
        "batch_shape": tuple(images.shape),
        "dtype": str(images.dtype),
        "min": round(float(images.min()), 4),
        "max": round(float(images.max()), 4),
        "mean": round(float(images.mean()), 4),
        "example_ids": list(ids[:3]),
        f"label_histogram_first_{n_label_batches}_batches": hist,
        "n_train_images": len(loaders.datasets["train"]),
        "n_val_images": len(loaders.datasets["val"]),
        "n_test_images": len(loaders.datasets["test"]),
    }

    if time_one_epoch:
        t0 = time.perf_counter()
        for _ in loaders.train:
            pass
        report["seconds_per_train_epoch"] = round(time.perf_counter() - t0, 2)
    return report


def compute_norm_stats(dataset: MILK10kImageDataset, n_images: int = 500,
                       seed: int = SEED) -> tuple[list[float], list[float]]:
    """Per-channel mean/std over a sample of the TRAIN split only (B8).

    Train-only is the point: computing normalisation statistics over the whole
    dataset leaks test-set information into training, in exactly the same way a
    scaler fitted before splitting does.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset))[:min(n_images, len(dataset))]
    total = np.zeros(3)
    total_sq = np.zeros(3)
    n_px = 0
    for i in idx:
        row = dataset.table.iloc[int(i)]
        img = _load_rgb(dataset.img_dir / f"{row['isic_id']}.jpg", str(row["isic_id"]))
        arr = np.asarray(img, dtype=np.float64).reshape(-1, 3) / 255.0
        total += arr.sum(axis=0)
        total_sq += (arr**2).sum(axis=0)
        n_px += len(arr)
    mean = total / n_px
    std = np.sqrt(np.maximum(total_sq / n_px - mean**2, 1e-12))
    return [round(float(m), 4) for m in mean], [round(float(s), 4) for s in std]


def save_json(obj: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    return path
