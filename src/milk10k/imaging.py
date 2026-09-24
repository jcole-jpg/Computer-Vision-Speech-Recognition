"""Pixel-level utilities: NumPy geometry, memory budgeting, JPEG quality (A2).

Written from scratch on purpose - the exercise is to understand what Pillow's
``transpose`` and a PSNR function actually do, not to call them.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------------------------------
# A2.1 - geometry with nothing but indexing
# --------------------------------------------------------------------------


def flip_lr(a: np.ndarray) -> np.ndarray:
    """Left-right flip: reverse the column axis. Returns a VIEW."""
    return a[:, ::-1]


def flip_ud(a: np.ndarray) -> np.ndarray:
    """Up-down flip: reverse the row axis. Returns a VIEW."""
    return a[::-1, :]


def transpose_hw(a: np.ndarray) -> np.ndarray:
    """Swap the height and width axes, leaving colour alone. Returns a VIEW."""
    return a.swapaxes(0, 1)


def rotate90_ccw(a: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees counter-clockwise = transpose, then flip up-down. VIEW."""
    return transpose_hw(a)[::-1, :]


def rotate90_cw(a: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees clockwise = transpose, then flip left-right. VIEW."""
    return transpose_hw(a)[:, ::-1]


#: name -> (our NumPy implementation, the Pillow transpose constant to check against)
NUMPY_VS_PIL = {
    "flip left-right": (flip_lr, Image.Transpose.FLIP_LEFT_RIGHT),
    "flip up-down": (flip_ud, Image.Transpose.FLIP_TOP_BOTTOM),
    "rotate 90 CCW": (rotate90_ccw, Image.Transpose.ROTATE_90),
    "transpose": (transpose_hw, Image.Transpose.TRANSPOSE),
}


def verify_against_pillow(img: Image.Image) -> pd.DataFrame:
    """Check each NumPy op against Pillow's transpose - arrays must be identical."""
    arr = np.asarray(img)
    rows = []
    for name, (fn, pil_op) in NUMPY_VS_PIL.items():
        ours = np.asarray(fn(arr))
        theirs = np.asarray(img.transpose(pil_op))
        rows.append({
            "operation": name,
            "our_shape": str(ours.shape),
            "pillow_shape": str(theirs.shape),
            "identical": bool(ours.shape == theirs.shape and np.array_equal(ours, theirs)),
            "is_view": bool(np.shares_memory(fn(arr), arr)),
        })
    return pd.DataFrame(rows)


def memory_budget(n_images: int, shapes: dict[str, tuple[int, int, int]]) -> pd.DataFrame:
    """RAM needed to hold ``n_images`` arrays of each shape, as uint8 and float32 (A2.1c)."""
    rows = []
    for name, shape in shapes.items():
        px = int(np.prod(shape))
        for dtype in (np.uint8, np.float32):
            nbytes = px * np.dtype(dtype).itemsize * n_images
            rows.append({
                "resolution": name,
                "shape": str(shape),
                "dtype": np.dtype(dtype).name,
                "bytes_per_image": px * np.dtype(dtype).itemsize,
                "total_GiB": round(nbytes / 1024**3, 2),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# A2.2 - JPEG quality and the file-size shortcut
# --------------------------------------------------------------------------


def psnr(original: np.ndarray, compressed: np.ndarray, max_value: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB, implemented from the definition.

    PSNR = 10 * log10(MAX^2 / MSE). Identical images have MSE 0, so PSNR is
    infinite - returned as ``inf`` rather than raising.
    """
    a = original.astype(np.float64)
    b = compressed.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(max_value**2 / mse))


@dataclass
class ReencodeResult:
    quality: int
    n_bytes: int
    psnr_db: float
    image: Image.Image


def reencode_jpeg(img: Image.Image, quality: int) -> ReencodeResult:
    """Re-encode in memory at a given JPEG quality and measure the damage."""
    original = np.asarray(img.convert("RGB"))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    n_bytes = buf.tell()
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    return ReencodeResult(quality, n_bytes, psnr(original, np.asarray(decoded)), decoded)


def quality_sweep(img: Image.Image,
                  qualities: tuple[int, ...] = (95, 75, 50, 25, 10)
                  ) -> tuple[pd.DataFrame, dict[int, Image.Image]]:
    """Re-encode one image across qualities; return the table and the decoded images."""
    results = [reencode_jpeg(img, q) for q in qualities]
    table = pd.DataFrame([{
        "quality": r.quality,
        "size_KB": round(r.n_bytes / 1024, 1),
        "psnr_dB": round(r.psnr_db, 2),
    } for r in results])
    return table, {r.quality: r.image for r in results}


def file_size_table(images: pd.DataFrame, path_col: str = "path") -> pd.DataFrame:
    """File size of every image via ``os.stat`` - no decoding at all (A2.2c)."""
    sizes = [os.stat(p).st_size for p in images[path_col]]
    return images.assign(n_bytes=sizes, size_KB=np.array(sizes) / 1024)


def file_size_auc(sized: pd.DataFrame, target: str = "diagnosis_1",
                  positive: str = "Malignant",
                  by: str = "image_type") -> pd.DataFrame:
    """ROC-AUC of raw file size as a malignancy score, per image type (A2.2c).

    An AUC far from 0.5 means the *bytes on disk* - before a single pixel is
    decoded - already separate the classes. That is an acquisition shortcut, not
    biology, and a CNN is perfectly capable of latching onto its visual cause.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    for group, g in sized.groupby(by):
        mask = g[target].isin([positive]) | ~g[target].isin([positive])
        g = g[mask]
        y = (g[target] == positive).astype(int)
        if y.nunique() < 2:
            continue
        rows.append({
            by: group,
            "n": len(g),
            "pct_positive": round(100 * y.mean(), 1),
            "median_KB_benign": round(g.loc[y == 0, "size_KB"].median(), 1),
            "median_KB_malignant": round(g.loc[y == 1, "size_KB"].median(), 1),
            "roc_auc_filesize": round(roc_auc_score(y, g["n_bytes"]), 4),
        })
    return pd.DataFrame(rows)


def detect_jpeg_quality(path: str | Path,
                        candidates: tuple[int, ...] = tuple(range(1, 101))) -> int | None:
    """Recover the JPEG quality a file was saved at, from its quantization table.

    A JPEG stores the quantization table it used, and Pillow generates a
    deterministic table per quality setting. Re-encoding a reference image at
    each candidate quality and matching the resulting table against the file's
    recovers the original setting exactly. Returns ``None`` if nothing matches
    (the encoder used custom tables).

    This matters for A2.2: re-encoding at the *same* quality reuses the same
    quantization lattice, so the coefficients are already on it and survive the
    round-trip nearly unchanged - which makes PSNR *peak* at the source quality
    instead of decreasing monotonically. A PSNR spike is therefore a fingerprint
    of the original encoder setting, not a measurement error.
    """
    with Image.open(path) as im:
        target = {k: list(v) for k, v in im.quantization.items()}
        size = im.size

    probe = Image.fromarray(np.zeros((min(size[1], 64), min(size[0], 64), 3), dtype=np.uint8))
    for q in candidates:
        buf = io.BytesIO()
        probe.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        with Image.open(buf) as ref:
            if {k: list(v) for k, v in ref.quantization.items()} == target:
                return q
    return None


def quantization_table_report(images: pd.DataFrame, n: int = 300,
                              path_col: str = "path") -> pd.DataFrame:
    """How many sampled images share each detected source JPEG quality (A2.2b)."""
    qualities = [detect_jpeg_quality(p) for p in images[path_col].head(n)]
    s = pd.Series(qualities, dtype="object").value_counts(dropna=False)
    return (s.rename("n_images").rename_axis("detected_source_quality")
             .reset_index().assign(pct=lambda d: (100 * d.n_images / len(qualities)).round(1)))
