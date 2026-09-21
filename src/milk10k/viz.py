"""Visualisation utilities (Part 5) plus the plots used by Parts 1-3.

Colour rules used throughout (see reports/ for the rationale):

* eleven classes are too many for eleven distinguishable hues, so class
  identity is carried by *position* (x-axis, panel) and colour encodes the
  one thing that matters clinically - **benign (blue) vs malignant (orange)**;
* where the assignment asks for eleven overlaid curves, the benign classes
  take steps of a blue ramp and the malignant ones steps of an orange ramp,
  with direct labels, and a small-multiples version sits next to it;
* magnitude tables (class share per category) use one sequential hue.

Part 5 entry points
-------------------
``show_image_grid(images_or_batch, labels=None, ...)``  grid of images titled with labels
``plot_class_balance(df_or_loader, ...)``                images per class, bar chart
``plot_batch_summary(batch, ...)``                       pixel-value distribution + raw/processed strip
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .data import (BENIGN_COLOR, MALIGNANT_COLOR, CLASS_CODES, CLASS_NAMES,
                   MALIGNANT, IMAGE_DIR)
from .preprocess import PreprocessConfig, preprocess_image

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BENIGN_CLASSES = [c for c in CLASS_CODES if c not in MALIGNANT]
MALIGNANT_CLASSES = [c for c in CLASS_CODES if c in MALIGNANT]


def apply_style() -> None:
    """Recessive chrome: thin axes, hairline grid, no top/right spines."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.labelcolor": INK2, "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
    })


def class_color(code: str) -> str:
    return MALIGNANT_COLOR if code in MALIGNANT else BENIGN_COLOR


def class_ramp_colors() -> dict[str, str]:
    """One distinct step per class: blues for benign, oranges for malignant."""
    blues = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(BENIGN_CLASSES)))
    oranges = plt.get_cmap("Oranges")(np.linspace(0.45, 0.95, len(MALIGNANT_CLASSES)))
    out = {c: mpl.colors.to_hex(col) for c, col in zip(BENIGN_CLASSES, blues)}
    out.update({c: mpl.colors.to_hex(col) for c, col in zip(MALIGNANT_CLASSES, oranges)})
    return out


def _family_legend(ax, loc="upper right"):
    handles = [plt.Line2D([], [], color=BENIGN_COLOR, lw=6, label="benign"),
               plt.Line2D([], [], color=MALIGNANT_COLOR, lw=6, label="malignant / pre-malignant")]
    ax.legend(handles=handles, loc=loc)


def _save(fig, save: str | Path | None):
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save)


# =============================================================================
# Part 5 - the three required utilities
# =============================================================================
def _to_display(img: np.ndarray) -> np.ndarray:
    """Any float/uint8 (H,W,C) array -> something imshow renders sensibly."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if lo >= 0.0 and hi <= 1.0:
        return arr
    if lo >= 0.0 and hi <= 255.0 and hi > 1.0:
        return arr / 255.0
    return (arr - lo) / (hi - lo + 1e-9)          # z-scored etc.: stretch for display only


def show_image_grid(images, labels: Sequence | None = None, *, ncols: int = 6,
                    max_images: int = 24, title: str | None = None,
                    figsize_per_cell: float = 1.9, save: str | Path | None = None):
    """Grid of images with their labels as titles.

    ``images`` may be a list/array of image arrays, a list of file paths, or a
    :class:`milk10k.loader.Batch` (then ``labels`` defaults to
    ``batch.label_names`` and the ``isic_id`` goes into the title too).
    """
    ids = None
    if hasattr(images, "label_names"):                 # a Batch
        batch = images
        images, ids = batch.images, batch.ids
        labels = labels if labels is not None else batch.label_names
    images = list(images)[:max_images]
    labels = list(labels)[:max_images] if labels is not None else [None] * len(images)
    n = len(images)
    ncols = min(ncols, max(n, 1))
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per_cell, nrows * figsize_per_cell * 1.08))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        img = images[i]
        if isinstance(img, (str, Path)):
            img = np.asarray(Image.open(img).convert("RGB"))
        ax.imshow(_to_display(img), cmap="gray" if np.ndim(_to_display(img)) == 2 else None)
        lab = labels[i]
        t = "" if lab is None else str(lab)
        if ids is not None:
            t = f"{t}\n{ids[i]}"
        color = class_color(str(lab)) if str(lab) in CLASS_NAMES else INK
        ax.set_title(t, fontsize=8, color=color, loc="center", fontweight="bold")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_class_balance(data, *, label_col: str = "label", title: str | None = None,
                       ax=None, save: str | Path | None = None):
    """Images per class as a bar chart. ``data`` is a DataFrame or a MILK10kLoader."""
    if hasattr(data, "class_counts"):                  # a loader
        counts = data.class_counts()
    else:
        counts = data[label_col].value_counts().reindex(
            [c for c in CLASS_CODES if c in set(data[label_col])], fill_value=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
    else:
        fig = ax.figure
    colors = [class_color(c) for c in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.72)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n{v / counts.sum():.1%}",
                ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_ylabel("images")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title(title or f"Class balance - {counts.sum():,} images, "
                          f"imbalance ratio {counts.max() / max(counts.min(), 1):.0f}:1")
    _family_legend(ax)
    ax.grid(axis="x", visible=False)
    _save(fig, save)
    return fig


def plot_batch_summary(batch, *, n_examples: int = 5, config: PreprocessConfig | None = None,
                       image_dir: Path | None = None, save: str | Path | None = None):
    """Debug view of one batch: pixel-value distribution + raw-vs-processed strip.

    Top row: per-channel histogram of the whole batch with its min/max/mean/std
    printed, so you can see at a glance whether normalisation did what the
    config says. Bottom rows: the first ``n_examples`` raw files (re-read from
    disk via ``batch.ids``) above their processed versions.
    """
    imgs = batch.images
    C = imgs.shape[-1]
    n_examples = min(n_examples, len(batch))
    fig = plt.figure(figsize=(1.9 * max(n_examples, 4), 6.4))
    gs = fig.add_gridspec(3, max(n_examples, 1), height_ratios=[1.6, 1, 1], hspace=0.55, wspace=0.08)

    ax = fig.add_subplot(gs[0, :])
    chan_colors = ["#d03b3b", "#0ca30c", "#2a78d6"] if C == 3 else [INK2]
    names = ["R", "G", "B"] if C == 3 else ["gray"]
    for c in range(C):
        vals = imgs[..., c].ravel()
        ax.hist(vals, bins=64, histtype="step", color=chan_colors[c], label=names[c], density=True)
    stats = (f"batch {imgs.shape}  dtype {imgs.dtype}   min {imgs.min():.3f}  max {imgs.max():.3f}  "
             f"mean {imgs.mean():.3f}  std {imgs.std():.3f}   skipped {len(batch.skipped)}")
    ax.set_title("Pixel-value distribution of the processed batch")
    ax.text(0.01, 0.96, stats, transform=ax.transAxes, fontsize=7.5, color=INK2,
            family="monospace", va="top", ha="left")
    ax.set_xlabel("value", labelpad=1); ax.set_ylabel("density")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    if C > 1:
        ax.legend(loc="upper right")

    image_dir = image_dir or IMAGE_DIR
    for j in range(n_examples):
        raw_path = Path(image_dir) / f"{batch.ids[j]}.jpg"
        ax_r = fig.add_subplot(gs[1, j]); ax_p = fig.add_subplot(gs[2, j])
        for a in (ax_r, ax_p):
            a.axis("off")
        if raw_path.is_file():
            raw = Image.open(raw_path).convert("RGB")
            ax_r.imshow(raw)
            ax_r.set_title(f"raw {raw.size[1]}x{raw.size[0]}\n{batch.label_names[j]}", fontsize=7.5,
                           color=class_color(batch.label_names[j]), loc="center")
        disp = _to_display(imgs[j])
        ax_p.imshow(disp, cmap="gray" if disp.ndim == 2 else None)
        ax_p.set_title(f"processed {imgs.shape[1]}x{imgs.shape[2]}x{C}", fontsize=7.5, color=INK2, loc="center")
    _save(fig, save)
    return fig


# =============================================================================
# Part 1 plots
# =============================================================================
def plot_categorical_association(result: dict, *, ax=None, annotate: bool = True,
                                 save: str | Path | None = None):
    """Heatmap of P(class | category) with the row counts; one sequential hue."""
    share, counts = result["class_share"], result["counts"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(share) + 1.3))
    else:
        fig = ax.figure
    im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=max(0.5, float(share.values.max())), aspect="auto")
    ax.set_xticks(range(share.shape[1]), share.columns, fontsize=8)
    ax.set_yticks(range(share.shape[0]),
                  [f"{r}  (n={counts.loc[r].sum():,})" for r in share.index], fontsize=8)
    for lab in ax.get_xticklabels():
        lab.set_color(class_color(lab.get_text()))
    if annotate:
        for i in range(share.shape[0]):
            for j in range(share.shape[1]):
                v = share.values[i, j]
                if v >= 0.02:
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.35 else INK)
    ax.grid(False)
    ax.set_title(f"{result['field']} -> class share per category   "
                 f"(χ² = {result['statistic']:.0f}, dof {result['dof']}, p = {result['p_value']:.1e}, "
                 f"Cramér's V = {result['effect_size']:.2f})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label("P(class | category)", fontsize=8)
    _save(fig, save)
    return fig


def plot_numeric_by_class(df: pd.DataFrame, field: str, result: dict | None = None, *,
                          label_col: str = "label", ax=None, save: str | Path | None = None):
    """Boxplots of a numeric field per class, coloured benign/malignant."""
    classes = [c for c in CLASS_CODES if c in set(df[label_col])]
    data = [df.loc[df[label_col] == c, field].dropna().to_numpy() for c in classes]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 3.0))
    else:
        fig = ax.figure
    bp = ax.boxplot(data, tick_labels=classes, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops={"color": INK, "lw": 1.2},
                    whiskerprops={"color": MUTED, "lw": 0.8}, capprops={"color": MUTED, "lw": 0.8})
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(class_color(c)); patch.set_alpha(0.55); patch.set_edgecolor("none")
    for i, d in enumerate(data):
        ax.text(i + 1, ax.get_ylim()[1], f"n={len(d)}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_ylabel(field)
    ax.grid(axis="x", visible=False)
    t = f"{field} by class"
    if result:
        t += (f"   (Kruskal-Wallis H = {result['statistic']:.0f}, p = {result['p_value']:.1e}, "
              f"ε² = {result['effect_size']:.3f})")
    ax.set_title(t, fontsize=9)
    _family_legend(ax, loc="upper left")
    _save(fig, save)
    return fig


def plot_association_ranking(summary: pd.DataFrame, *, save: str | Path | None = None):
    """Horizontal bars of effect size per field, one colour per statistic type."""
    s = summary.sort_values("effect_r")
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(s) + 1.2))
    colors = [BENIGN_COLOR if t == "chi-square" else MALIGNANT_COLOR for t in s["test"]]
    ax.barh(s["field"], s["effect_r"], color=colors, height=0.66)
    for y, (v, e, name, note, miss) in enumerate(zip(s["effect_r"], s["effect_size"], s["effect_name"],
                                                      s["note"], s["pct_missing"])):
        raw = f"  (ε² = {e:.2f})" if name == "epsilon²" else ""
        tag = ("  · bias risk" if str(note).startswith("BIAS") else "") + (f"  · {miss:.0f}% missing" if miss >= 5 else "")
        ax.text(v + 0.005, y, f"{v:.2f}{raw}{tag}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, max(0.8, s["effect_r"].max() * 1.45))
    ax.set_xlabel("effect size on a common scale  (Cramér's V for categorical · √ε² for numeric)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=BENIGN_COLOR, lw=6, label="categorical / boolean -> chi-square + Cramér's V"),
               plt.Line2D([], [], color=MALIGNANT_COLOR, lw=6, label="numeric / ordinal -> Kruskal-Wallis + ε²")]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("Association of each metadata field with the 11-class label")
    _save(fig, save)
    return fig


# =============================================================================
# Part 2 plots
# =============================================================================
def plot_class_histograms(hists: dict, centers: np.ndarray, *, modality: str = "",
                          save: str | Path | None = None):
    """Overlaid average histograms: grayscale (left) + R, G, B (right three)."""
    ramp = class_ramp_colors()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharey=True)
    panels = [("grayscale", lambda h: h["gray"]), ("R channel", lambda h: h["rgb"][0]),
              ("G channel", lambda h: h["rgb"][1]), ("B channel", lambda h: h["rgb"][2])]
    for ax, (name, pick) in zip(axes, panels):
        for cls, h in hists.items():
            ax.plot(centers, pick(h), color=ramp[cls], lw=1.4,
                    ls="-" if cls in MALIGNANT else "--", label=f"{cls} (n={h['n']})")
        ax.set_title(f"{name} - class-average histogram", fontsize=9)
        ax.set_xlim(0, 255); ax.set_xlabel("intensity")
    axes[0].set_ylabel("density")
    axes[-1].legend(ncols=1, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"{modality} images: benign = blues (dashed), malignant = oranges (solid)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_class_histogram_grid(hists: dict, centers: np.ndarray, *, modality: str = "",
                              save: str | Path | None = None):
    """Small multiples: each class's grayscale + RGB histogram vs the all-class mean."""
    classes = list(hists)
    ncols = 4
    nrows = -(-len(classes) // ncols)
    all_gray = np.mean([h["gray"] for h in hists.values()], axis=0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 2.3 * nrows), sharex=True, sharey=True)
    for ax, cls in zip(axes.ravel(), classes):
        h = hists[cls]
        ax.plot(centers, all_gray, color=MUTED, lw=1, ls=":", label="all classes (gray)")
        ax.plot(centers, h["gray"], color=INK, lw=1.4, label="gray")
        for c, col in zip(range(3), ["#d03b3b", "#0ca30c", "#2a78d6"]):
            ax.plot(centers, h["rgb"][c], color=col, lw=0.9, alpha=0.85, label="RGB"[c])
        short = CLASS_NAMES[cls] if len(CLASS_NAMES[cls]) <= 34 else CLASS_NAMES[cls][:32] + "…"
        ax.set_title(f"{cls}  (n={h['n']})\n{short}", fontsize=8, color=class_color(cls))
        ax.set_xlim(0, 255)
    for ax in axes.ravel()[len(classes):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle(f"{modality} images: per-class histograms (black = gray, R/G/B channels, dotted = all-class mean)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_color_boxplots(table: pd.DataFrame, *, label_col: str = "label",
                        modality: str = "", save: str | Path | None = None):
    """Per-class boxplots of grayscale mean, R/G/B means and grayscale std."""
    cols = [("gray_mean", "grayscale mean"), ("r_mean", "R mean"), ("g_mean", "G mean"),
            ("b_mean", "B mean"), ("gray_std", "grayscale std (contrast)")]
    classes = [c for c in CLASS_CODES if c in set(table[label_col])]
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 3.0))
    for ax, (col, name) in zip(axes, cols):
        data = [table.loc[table[label_col] == c, col].to_numpy() for c in classes]
        bp = ax.boxplot(data, tick_labels=classes, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": INK, "lw": 1.1},
                        whiskerprops={"color": MUTED, "lw": 0.8}, capprops={"color": MUTED, "lw": 0.8})
        for patch, c in zip(bp["boxes"], classes):
            patch.set_facecolor(class_color(c)); patch.set_alpha(0.55); patch.set_edgecolor("none")
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("0-255")
    fig.suptitle(f"{modality} images: per-image colour statistics by class (blue = benign, orange = malignant)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, save)
    return fig


# =============================================================================
# Part 3 plot
# =============================================================================
def plot_before_after(paths: Sequence, configs: dict[str, PreprocessConfig], *,
                      labels: Sequence[str] | None = None, save: str | Path | None = None):
    """Raw image (left) next to its output under each named config (columns)."""
    n = len(paths)
    ncol = 1 + len(configs)
    fig, axes = plt.subplots(n, ncol, figsize=(2.1 * ncol, 2.25 * n))
    axes = np.atleast_2d(axes)
    for i, p in enumerate(paths):
        raw = Image.open(p).convert("RGB")
        axes[i, 0].imshow(raw)
        lab = f"{labels[i]}\n" if labels is not None else "\n"
        axes[i, 0].set_title(f"{lab}raw\n{raw.size[1]}x{raw.size[0]}x3 uint8 [0, 255]", fontsize=7.5, loc="center")
        for j, (name, cfg) in enumerate(configs.items(), start=1):
            arr, info = preprocess_image(p, cfg)
            disp = _to_display(arr)
            axes[i, j].imshow(disp, cmap="gray" if disp.ndim == 2 else None)
            axes[i, j].set_title(f"{name}\n{'x'.join(map(str, info.output_shape))} "
                                 f"[{info.value_min:.2f}, {info.value_max:.2f}]", fontsize=7, loc="center")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    _save(fig, save)
    return fig
