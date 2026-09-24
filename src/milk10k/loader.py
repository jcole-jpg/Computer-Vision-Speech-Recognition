"""Batch data loader (Part 4): (processed image, label) batches, lazily.

Interface
---------
Construct with a metadata table and a preprocessing config::

    df = milk10k.load_images_table()
    loader = MILK10kLoader(df, PreprocessConfig(size=(128, 128)), batch_size=32,
                           shuffle=True, seed=0)

* rows whose image file is missing on disk are dropped at construction
  (``loader.df`` is the surviving table, ``loader.n_unavailable`` how many went)
* ``len(loader)`` is the number of batches per epoch
* ``for batch in loader`` yields :class:`Batch` objects with

    - ``batch.images``  float32 ``(B, H, W, C)`` processed on the fly
    - ``batch.labels``  int64 ``(B,)`` class indices into ``loader.classes``
    - ``batch.ids``     list of ``isic_id`` strings in the same order
    - ``batch.skipped`` ``(id, reason)`` for rows that failed in this batch

  ``batch.label_names`` maps the indices back to class codes.
* ``loader.classes`` is the ordered list of class codes (index = label).
* Nothing is loaded until you iterate, so the table can be all 10,480 rows.

Reshuffle happens at the start of every epoch when ``shuffle=True``;
pass ``seed`` for reproducible order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from .data import CLASS_CODES, IMAGE_DIR, available_subset
from .preprocess import PreprocessConfig, preprocess_batch


@dataclass
class Batch:
    images: np.ndarray            # (B, H, W, C) float32
    labels: np.ndarray            # (B,) int64
    ids: list[str]
    skipped: list[tuple[str, str]]
    classes: Sequence[str]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def label_names(self) -> list[str]:
        return [self.classes[i] for i in self.labels]


class MILK10kLoader:
    """Iterable over batches of (processed images, integer labels)."""

    def __init__(
        self,
        df: pd.DataFrame,
        config: PreprocessConfig | None = None,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        label_col: str = "label",
        classes: Sequence[str] | None = None,
        image_dir: Path | None = None,
        drop_last: bool = False,
        strict: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if label_col not in df.columns:
            raise ValueError(f"label column {label_col!r} not in DataFrame")

        self.config = config or PreprocessConfig()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.label_col = label_col
        self._rng = np.random.default_rng(seed)

        # Images that are not on disk. By default they are dropped (the Session 2
        # behaviour, useful when working from a partial download); with
        # ``strict=True`` they raise instead, which is what the project pipeline
        # uses - silently skipping rows shrinks the dataset and shifts the class
        # balance without ever warning you (B1).
        self.strict = strict
        n_before = len(df)
        df = df.dropna(subset=[label_col])
        available = available_subset(df, image_dir or IMAGE_DIR)
        self.n_unavailable = n_before - len(available)
        if strict and self.n_unavailable:
            missing = sorted(set(df["isic_id"]) - set(available["isic_id"]))[:10]
            raise FileNotFoundError(
                f"{self.n_unavailable} of {n_before} rows have no image on disk "
                f"(first few: {missing}). Pass strict=False to skip them instead."
            )
        df = available
        if df.empty:
            raise ValueError("no rows with both a label and an image file on disk")

        self.classes: list[str] = list(classes) if classes is not None else [
            c for c in CLASS_CODES if c in set(df[label_col])
        ] or sorted(df[label_col].unique())
        unknown = set(df[label_col]) - set(self.classes)
        if unknown:
            raise ValueError(f"labels not in classes: {sorted(unknown)}")
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}
        self.df = df.reset_index(drop=True)
        self._labels = self.df[label_col].map(self.class_to_index).to_numpy(dtype=np.int64)

    # -- sizes ---------------------------------------------------------------
    @property
    def n_images(self) -> int:
        return len(self.df)

    def __len__(self) -> int:
        n = self.n_images
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    # -- iteration -------------------------------------------------------------
    def _order(self) -> np.ndarray:
        idx = np.arange(self.n_images)
        return self._rng.permutation(idx) if self.shuffle else idx

    def get_batch(self, indices: Sequence[int]) -> Batch:
        """Process the rows at ``indices`` (positions in ``loader.df``)."""
        rows = self.df.iloc[list(indices)]
        result = preprocess_batch(rows[["isic_id", "path"]], self.config)
        kept = {i for i in result.ids}
        labels = np.array(
            [self._labels[i] for i, isic in zip(indices, rows["isic_id"]) if isic in kept],
            dtype=np.int64,
        )
        return Batch(images=result.images, labels=labels, ids=result.ids,
                     skipped=result.skipped, classes=self.classes)

    def __iter__(self) -> Iterator[Batch]:
        order = self._order()
        n_batches = len(self)
        for b in range(n_batches):
            yield self.get_batch(order[b * self.batch_size:(b + 1) * self.batch_size])

    def class_counts(self) -> pd.Series:
        """Images per class in the available data, in ``classes`` order."""
        counts = self.df[self.label_col].value_counts()
        return counts.reindex(self.classes, fill_value=0)

    def __repr__(self) -> str:
        return (f"MILK10kLoader(n_images={self.n_images}, batches={len(self)}, "
                f"batch_size={self.batch_size}, classes={len(self.classes)}, "
                f"shape={self.config.output_shape}, unavailable={self.n_unavailable})")
