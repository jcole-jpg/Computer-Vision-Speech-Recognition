"""Dataset-level colour and histogram analysis (Part 2).

Extends the single-image histogram of Session 2 to a class comparison:

* :func:`sample_per_class`     - a seeded, balanced sample (``n`` per class
  *per modality*; the dermoscopic / clinical split dominates colour, so the
  two modalities must never be pooled when comparing classes)
* :func:`image_color_stats`    - per-image RGB / grayscale mean, std and
  histograms, computed from the *raw* image (no normalisation, so the
  numbers are on the familiar 0-255 scale)
* :func:`class_histograms`     - the average grayscale and per-channel RGB
  histogram of every class
* :func:`class_color_summary`  - per-class mean / std table of the summary stats

All heavy lifting goes through :func:`milk10k.preprocess.preprocess_image`
with ``normalize=None`` so the same resize / decode path is used everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CLASS_CODES
from .preprocess import PreprocessConfig, preprocess_image

N_BINS = 64
CHANNELS = ("R", "G", "B")
# raw statistics are computed on a modest resize: histograms and means are
# practically identical to full-res, and 660 images load in seconds
_STATS_CFG = PreprocessConfig(size=(256, 256), color_mode="rgb", resize_mode="crop",
                              normalize=None)


def sample_per_class(images: pd.DataFrame, n_per_class: int = 30, *,
                     seed: int = 42, label_col: str = "label",
                     modality_col: str = "image_type") -> pd.DataFrame:
    """Up to ``n_per_class`` rows per (class, modality), fixed seed.

    Classes with fewer than ``n_per_class`` images keep everything they have
    (MAL_OTH has only 9 lesions), which is reported via ``value_counts``.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for (_, _), grp in images.groupby([label_col, modality_col]):
        k = min(n_per_class, len(grp))
        parts.append(grp.iloc[rng.choice(len(grp), size=k, replace=False)])
    return (pd.concat(parts).sort_values([label_col, modality_col, "isic_id"])
            .reset_index(drop=True))


@dataclass
class ImageColorStats:
    isic_id: str
    gray_mean: float
    gray_std: float
    r_mean: float
    g_mean: float
    b_mean: float
    r_std: float
    g_std: float
    b_std: float
    gray_hist: np.ndarray       # (N_BINS,) density
    rgb_hist: np.ndarray        # (3, N_BINS) density


def image_color_stats(path, n_bins: int = N_BINS) -> ImageColorStats:
    """Raw 0-255 RGB + grayscale summary statistics and histograms for one image."""
    rgb, _ = preprocess_image(path, _STATS_CFG)                     # (H, W, 3) float32 0-255
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    edges = np.linspace(0, 256, n_bins + 1)
    gray_hist, _ = np.histogram(gray, bins=edges, density=True)
    rgb_hist = np.stack([np.histogram(rgb[..., c], bins=edges, density=True)[0]
                         for c in range(3)])
    means, stds = rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))
    return ImageColorStats(
        isic_id=str(path).rsplit("/", 1)[-1].removesuffix(".jpg"),
        gray_mean=float(gray.mean()), gray_std=float(gray.std()),
        r_mean=float(means[0]), g_mean=float(means[1]), b_mean=float(means[2]),
        r_std=float(stds[0]), g_std=float(stds[1]), b_std=float(stds[2]),
        gray_hist=gray_hist, rgb_hist=rgb_hist,
    )


def compute_color_table(sample: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Per-image colour stats for every row of ``sample`` (needs ``path``).

    Returns ``sample`` joined with the summary columns plus two object columns
    ``gray_hist`` / ``rgb_hist`` holding the per-image histograms. Unreadable
    files are dropped and listed in ``result.attrs["skipped"]``.
    """
    rows, skipped = [], []
    it = sample.itertuples(index=False)
    if verbose:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(sample), desc="colour stats")
        except ImportError:
            pass
    for row in it:
        try:
            st = image_color_stats(row.path)
        except Exception as exc:                                    # noqa: BLE001
            skipped.append((row.isic_id, f"{type(exc).__name__}: {exc}"))
            continue
        rows.append({**st.__dict__, "isic_id": row.isic_id})
    stats = pd.DataFrame(rows)
    out = sample.merge(stats, on="isic_id", how="inner")
    out.attrs["skipped"] = skipped
    return out


SUMMARY_COLS = ["gray_mean", "gray_std", "r_mean", "g_mean", "b_mean",
                "r_std", "g_std", "b_std"]


def class_histograms(table: pd.DataFrame, label_col: str = "label") -> dict[str, dict[str, np.ndarray]]:
    """``{class: {"gray": (N_BINS,), "rgb": (3, N_BINS), "n": int}}`` averaged per class."""
    out = {}
    for cls, grp in table.groupby(label_col):
        out[cls] = {
            "gray": np.mean(np.stack(grp["gray_hist"].to_list()), axis=0),
            "rgb": np.mean(np.stack(grp["rgb_hist"].to_list()), axis=0),
            "n": int(len(grp)),
        }
    return {c: out[c] for c in CLASS_CODES if c in out}


def class_color_summary(table: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Per-class mean (and std across images) of the per-image colour statistics."""
    agg = table.groupby(label_col)[SUMMARY_COLS].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.insert(0, "n_images", table.groupby(label_col).size())
    return agg.reindex([c for c in CLASS_CODES if c in agg.index]).round(1)


def bin_centers(n_bins: int = N_BINS) -> np.ndarray:
    edges = np.linspace(0, 256, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2
