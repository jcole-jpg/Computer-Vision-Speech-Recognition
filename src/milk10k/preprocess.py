"""Reusable image preprocessing (Part 3): one image, then a batch of images.

The single-image function does three configurable things, in this order:

1. **resize** to ``config.size`` — either ``"crop"`` (aspect-preserving centre
   crop to the target aspect ratio, then resize; no distortion) or
   ``"stretch"`` (plain resize; keeps every pixel but distorts geometry).
2. **colour conversion** — ``"rgb"`` (H, W, 3) or ``"gray"`` (H, W, 1).
3. **normalisation** —
   ``"minmax"`` scales uint8 0–255 to float32 0–1 with the *global* constant
   255 (not the per-image min/max). Per-image min–max would stretch every
   image to full contrast and erase exactly the brightness / colour
   differences between modalities and classes that the EDA measures, so the
   global version is the default.
   ``"zscore"`` subtracts a mean and divides by a std, per channel. If
   ``config.mean``/``config.std`` are not given the *image's own* statistics
   are used (output mean 0, std 1). For model training pass dataset-level
   values (``mean=(r, g, b)``) so all images share one scale.
   ``None`` leaves the array as float32 in 0–255.

Every call returns the array **and** an :class:`ImageInfo` record describing
what was done (input/output shape, dtype, value range) so a caller never has
to guess what the array contains.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

ColorMode = Literal["rgb", "gray"]
ResizeMode = Literal["crop", "stretch"]
Normalize = Literal["minmax", "zscore"] | None


@dataclass(frozen=True)
class PreprocessConfig:
    """Everything the pipeline needs to know; immutable so it can be shared."""

    size: tuple[int, int] = (224, 224)      # (height, width)
    color_mode: ColorMode = "rgb"
    resize_mode: ResizeMode = "crop"
    normalize: Normalize = "minmax"
    mean: Sequence[float] | None = None      # only used for z-score
    std: Sequence[float] | None = None       # only used for z-score

    def __post_init__(self) -> None:
        if len(self.size) != 2 or min(self.size) < 1:
            raise ValueError(f"size must be (height, width) > 0, got {self.size}")
        if self.color_mode not in ("rgb", "gray"):
            raise ValueError(f"color_mode must be 'rgb' or 'gray', got {self.color_mode!r}")
        if self.resize_mode not in ("crop", "stretch"):
            raise ValueError(f"resize_mode must be 'crop' or 'stretch', got {self.resize_mode!r}")
        if self.normalize not in ("minmax", "zscore", None):
            raise ValueError(f"normalize must be 'minmax', 'zscore' or None, got {self.normalize!r}")
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must be given together")

    @property
    def channels(self) -> int:
        return 3 if self.color_mode == "rgb" else 1

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (self.size[0], self.size[1], self.channels)


@dataclass
class ImageInfo:
    """What happened to one image. ``source`` is the path or ``"<array>"``."""

    source: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str
    value_min: float
    value_max: float
    value_mean: float
    value_std: float
    steps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _to_pil(src: str | Path | np.ndarray | Image.Image) -> tuple[Image.Image, str, tuple[int, ...]]:
    """Open anything we accept as an RGB PIL image; return (image, source, input_shape)."""
    if isinstance(src, Image.Image):
        return src.convert("RGB"), "<PIL.Image>", (src.height, src.width, len(src.getbands()))
    if isinstance(src, np.ndarray):
        arr = src
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
            raise ValueError(f"array must be (H, W) or (H, W, 1|3|4), got shape {arr.shape}")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            # accept float images in 0–1 or 0–255
            arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr[..., :3]), "<array>", tuple(src.shape)
    path = Path(src)
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    with Image.open(path) as im:
        im.load()                               # force decode so errors surface here
        rgb = im.convert("RGB")
    return rgb, str(path), (im.height, im.width, len(im.getbands()))


def _resize(img: Image.Image, size: tuple[int, int], mode: ResizeMode) -> Image.Image:
    h, w = size
    if mode == "stretch":
        return img.resize((w, h), Image.BILINEAR)
    # centre crop to the target aspect ratio, then resize (no distortion)
    src_w, src_h = img.size
    target_ratio, src_ratio = w / h, src_w / src_h
    if src_ratio > target_ratio:          # too wide -> trim left/right
        new_w = int(round(src_h * target_ratio))
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    elif src_ratio < target_ratio:        # too tall -> trim top/bottom
        new_h = int(round(src_w / target_ratio))
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    return img.resize((w, h), Image.BILINEAR)


def preprocess_image(
    src: str | Path | np.ndarray | Image.Image,
    config: PreprocessConfig | None = None,
) -> tuple[np.ndarray, ImageInfo]:
    """Resize, convert colour space and normalise one image.

    Returns ``(array, info)`` where ``array`` is float32 of shape
    ``(H, W, C)`` — ``C`` = 3 for RGB, 1 for grayscale — and ``info`` records
    the input/output shape and the value range so the result is self-describing.
    Raises ``FileNotFoundError`` / ``UnidentifiedImageError`` / ``ValueError``
    for unreadable inputs; the batch function turns those into skips.
    """
    cfg = config or PreprocessConfig()
    img, source, input_shape = _to_pil(src)
    steps: list[str] = []

    img = _resize(img, cfg.size, cfg.resize_mode)
    steps.append(f"resize[{cfg.resize_mode}]->{cfg.size[0]}x{cfg.size[1]}")

    if cfg.color_mode == "gray":
        img = img.convert("L")              # ITU-R 601-2 luma: 0.299 R + 0.587 G + 0.114 B
        arr = np.asarray(img, dtype=np.float32)[..., None]
        steps.append("rgb->gray(luma)")
    else:
        arr = np.asarray(img, dtype=np.float32)
        steps.append("rgb")

    if cfg.normalize == "minmax":
        arr = arr / 255.0
        steps.append("minmax(/255)->[0,1]")
    elif cfg.normalize == "zscore":
        if cfg.mean is not None:
            mean = np.asarray(cfg.mean, dtype=np.float32).reshape(1, 1, -1)
            std = np.asarray(cfg.std, dtype=np.float32).reshape(1, 1, -1)
            steps.append("zscore(dataset mean/std)")
        else:
            mean = arr.mean(axis=(0, 1), keepdims=True)
            std = arr.std(axis=(0, 1), keepdims=True)
            steps.append("zscore(per-image mean/std)")
        arr = (arr - mean) / np.maximum(std, 1e-6)
    else:
        steps.append("no-normalise(float32 0-255)")

    arr = np.ascontiguousarray(arr, dtype=np.float32)
    info = ImageInfo(
        source=source,
        input_shape=input_shape,
        output_shape=tuple(arr.shape),
        dtype=str(arr.dtype),
        value_min=float(arr.min()),
        value_max=float(arr.max()),
        value_mean=float(arr.mean()),
        value_std=float(arr.std()),
        steps=steps,
    )
    return arr, info


@dataclass
class BatchResult:
    """Output of :func:`preprocess_batch`.

    ``images`` is a stacked ``(N, H, W, C)`` float32 array (all outputs share
    the config's shape, so stacking is always possible). ``ids`` are the
    identifiers of the images that made it, in order; ``skipped`` lists the
    ``(id, reason)`` of every input that failed.
    """

    images: np.ndarray
    ids: list[str]
    infos: list[ImageInfo]
    skipped: list[tuple[str, str]]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)


def _iter_items(items) -> Iterable[tuple[str, object]]:
    """Yield (id, source) pairs from a DataFrame, a mapping, or a plain sequence."""
    if hasattr(items, "iterrows"):                       # pandas DataFrame
        id_col = "isic_id" if "isic_id" in items.columns else None
        src_col = "path" if "path" in items.columns else None
        if src_col is None:
            raise ValueError("DataFrame must have a 'path' column (see load_images_table)")
        for i, row in items.iterrows():
            yield (str(row[id_col]) if id_col else str(row[src_col])), row[src_col]
    elif isinstance(items, dict):
        yield from items.items()
    else:
        for src in items:
            yield (str(src) if not isinstance(src, np.ndarray) else f"<array#{id(src)}>"), src


def preprocess_batch(
    items,
    config: PreprocessConfig | None = None,
    *,
    verbose: bool = False,
) -> BatchResult:
    """Apply :func:`preprocess_image` to many inputs, skipping the ones that fail.

    ``items`` may be a list of paths/arrays, a ``{id: path}`` mapping, or a
    DataFrame with ``path`` (and optionally ``isic_id``) columns. A missing or
    corrupt file never aborts the batch: it is recorded in ``result.skipped``
    with the exception message and processing continues.
    """
    cfg = config or PreprocessConfig()
    arrays, ids, infos, skipped = [], [], [], []
    for item_id, src in _iter_items(items):
        try:
            arr, info = preprocess_image(src, cfg)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append((item_id, reason))
            if verbose:
                print(f"skip {item_id}: {reason}")
            continue
        arrays.append(arr)
        ids.append(item_id)
        infos.append(info)
    images = (np.stack(arrays) if arrays
              else np.empty((0, *cfg.output_shape), dtype=np.float32))
    return BatchResult(images=images, ids=ids, infos=infos, skipped=skipped)
