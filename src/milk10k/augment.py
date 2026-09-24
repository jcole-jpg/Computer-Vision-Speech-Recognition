"""Transforms for the project pipeline (B7) and a quantitative safety audit (A3.5).

The transforms live here, in an importable module, rather than in a notebook, so
that every later milestone trains and evaluates with exactly the same pixels.

The audit exists because "rotation is fine, colour jitter is risky" is an
opinion. Colour *is* diagnostic signal in dermoscopy: if a jitter moves an image
further in hue than the average gap between Benign and Malignant, the
augmentation can turn one class into the other, and no amount of good intent
makes that safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

from .config import CONFIG

# --------------------------------------------------------------------------
# B7 - the pipeline transforms
# --------------------------------------------------------------------------

#: Every augmentation, its parameters, and why it cannot change the diagnosis.
#: This table is the deliverable for B7 - it is printed into the report.
AUGMENTATION_RATIONALE = pd.DataFrame([
    {"augmentation": "RandomResizedCrop", "parameters": "size=224, scale=(0.8, 1.0)",
     "medical_justification": "A lesion keeps its diagnosis when framed slightly "
                              "tighter or looser; scale is capped at 0.8 so the "
                              "lesion border - a key diagnostic feature - stays in frame."},
    {"augmentation": "RandomHorizontalFlip", "parameters": "p=0.5",
     "medical_justification": "Skin lesions have no canonical left-right orientation; "
                              "a mirrored mole is the same mole."},
    {"augmentation": "RandomVerticalFlip", "parameters": "p=0.5",
     "medical_justification": "Dermoscopy has no fixed 'up'; the camera angle is arbitrary."},
    {"augmentation": "RandomRotation", "parameters": "degrees=30",
     "medical_justification": "Rotation is a pure camera-orientation change. Asymmetry "
                              "(the 'A' of ABCDE) is rotation-invariant, so it is preserved."},
    {"augmentation": "ColorJitter (brightness/contrast)", "parameters": "brightness=0.1, contrast=0.1",
     "medical_justification": "Mimics real illumination and exposure variation between "
                              "clinics. Kept small - see A3.5: beyond ~0.3 the shift "
                              "exceeds the real Benign/Malignant brightness gap."},
    {"augmentation": "ColorJitter (hue)", "parameters": "hue=0.02",
     "medical_justification": "Absorbs white-balance differences only. Capped at 0.02 "
                              "(~7 degrees) because pigment colour is diagnostic signal; "
                              "larger values recolour the lesion itself."},
    {"augmentation": "Normalize", "parameters": "train-split mean/std",
     "medical_justification": "An affine rescale of intensities, identical in train and "
                              "eval; carries no label information."},
])


def train_transform(image_size: int = CONFIG.image_size,
                    mean: tuple[float, ...] = CONFIG.norm_mean,
                    std: tuple[float, ...] = CONFIG.norm_std) -> T.Compose:
    """Augmenting transform - TRAIN ONLY.

    Geometry is augmented freely (a lesion is orientation-free); colour is
    augmented only within the limits the A3.5 audit showed to be label-safe.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def eval_transform(image_size: int = CONFIG.image_size,
                   mean: tuple[float, ...] = CONFIG.norm_mean,
                   std: tuple[float, ...] = CONFIG.norm_std) -> T.Compose:
    """Deterministic transform for val and test.

    No random op anywhere: applying it twice to the same image must give
    bit-identical tensors, which :func:`assert_eval_deterministic` proves.
    Resize-then-centre-crop rather than a plain resize, so the aspect ratio is
    not distorted (MILK10k is 600x450, i.e. 4:3).
    """
    return T.Compose([
        T.Resize(int(image_size * 1.14), antialias=True),  # 224 -> 256
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def assert_eval_deterministic(img: Image.Image, transform: T.Compose | None = None) -> bool:
    """Apply the eval transform twice and assert the tensors are equal (B7)."""
    tf = transform if transform is not None else eval_transform()
    a, b = tf(img), tf(img)
    if not torch.equal(a, b):
        raise AssertionError(
            f"eval_transform is not deterministic: max abs diff {(a - b).abs().max():.3e}"
        )
    return True


# --------------------------------------------------------------------------
# A3.5 - is this augmentation label-safe?
# --------------------------------------------------------------------------


def circular_mean_degrees(hue_degrees: np.ndarray) -> float:
    """Mean of angles, done properly.

    Hue is circular: 359 degrees and 1 degree are 2 degrees apart, but their
    arithmetic mean is 180 - the opposite colour. Averaging the unit vectors and
    taking the angle back avoids that.
    """
    rad = np.deg2rad(np.asarray(hue_degrees, dtype=np.float64))
    ang = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
    deg = float(np.rad2deg(ang) % 360.0)
    # arctan2 returns a tiny NEGATIVE float for an angle that is mathematically
    # exactly 0 (e.g. the mean of 359 and 1), and -1e-15 % 360 wraps to 360.0 -
    # outside the documented [0, 360) range. Snap it back to 0.
    return 0.0 if deg >= 360.0 - 1e-9 else deg


def circular_gap_degrees(a: float, b: float) -> float:
    """Shortest angular distance between two hues, in [0, 180]."""
    d = abs(a - b) % 360.0
    return float(min(d, 360.0 - d))


def _center_patch(img: Image.Image, frac: float = 0.5) -> np.ndarray:
    """Central crop as a stand-in for a lesion mask.

    MILK10k ships no segmentation masks, and the lesion is framed centrally in
    both views, so the central half is a reasonable proxy for 'lesion pixels'.
    Stated explicitly because it is an assumption, not a measurement.
    """
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return np.asarray(img.convert("RGB").crop((left, top, left + cw, top + ch)))


def hsv_stats(img: Image.Image, frac: float = 0.5) -> tuple[float, float]:
    """(mean hue in degrees via circular mean, mean V in [0,1]) of the central patch."""
    patch = Image.fromarray(_center_patch(img, frac)).convert("HSV")
    arr = np.asarray(patch, dtype=np.float64)
    hue_deg = arr[..., 0] * (360.0 / 255.0)
    value = arr[..., 2] / 255.0
    return circular_mean_degrees(hue_deg.ravel()), float(value.mean())


def class_color_gap(images: pd.DataFrame, n_per_group: int = 100,
                    target: str = "diagnosis_1",
                    groups: tuple[str, str] = ("Benign", "Malignant"),
                    seed: int = 0, path_col: str = "path") -> tuple[pd.DataFrame, dict]:
    """Mean hue and brightness per class, and the gap between them (A3.5a)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in groups:
        pool = images[images[target] == g]
        take = pool.iloc[rng.permutation(len(pool))[:n_per_group]]
        for p in take[path_col]:
            with Image.open(p) as im:
                h, v = hsv_stats(im)
            rows.append({"group": g, "hue_deg": h, "value": v})

    df = pd.DataFrame(rows)
    summary = df.groupby("group").agg(
        n=("hue_deg", "size"),
        mean_hue_deg=("hue_deg", lambda s: circular_mean_degrees(s.to_numpy())),
        mean_value=("value", "mean"),
    ).round(4)
    gap = {
        "hue_gap_deg": round(circular_gap_degrees(
            summary.loc[groups[0], "mean_hue_deg"], summary.loc[groups[1], "mean_hue_deg"]), 3),
        "value_gap": round(abs(summary.loc[groups[0], "mean_value"]
                               - summary.loc[groups[1], "mean_value"]), 4),
    }
    return summary, gap


def jitter_shift(images: pd.DataFrame, hue_params=(0.02, 0.05, 0.1, 0.5),
                 brightness_params=(0.1, 0.3, 0.6), n_images: int = 40,
                 seed: int = 0, path_col: str = "path") -> pd.DataFrame:
    """How far does each ColorJitter setting actually move hue / brightness? (A3.5a)

    Each setting is applied at its extreme (torchvision samples uniformly in
    ``[-hue, +hue]`` and ``[1-b, 1+b]``), because the question is what the
    augmentation *can* do, not what it does on average.
    """
    rng = np.random.default_rng(seed)
    take = images.iloc[rng.permutation(len(images))[:n_images]]
    rows = []

    for hue in hue_params:
        shifts = []
        for p in take[path_col]:
            with Image.open(p) as im:
                im = im.convert("RGB")
                base_h, _ = hsv_stats(im)
                jittered = T.functional.adjust_hue(im, hue)  # extreme of the range
                new_h, _ = hsv_stats(jittered)
            shifts.append(circular_gap_degrees(base_h, new_h))
        rows.append({"parameter": "hue", "value": hue,
                     "nominal_shift_deg": round(hue * 360, 1),
                     "measured_hue_shift_deg": round(float(np.mean(shifts)), 3),
                     "measured_value_shift": np.nan})

    for b in brightness_params:
        shifts = []
        for p in take[path_col]:
            with Image.open(p) as im:
                im = im.convert("RGB")
                _, base_v = hsv_stats(im)
                jittered = T.functional.adjust_brightness(im, 1 + b)  # extreme
                _, new_v = hsv_stats(jittered)
            shifts.append(abs(new_v - base_v))
        rows.append({"parameter": "brightness", "value": b,
                     "nominal_shift_deg": np.nan,
                     "measured_hue_shift_deg": np.nan,
                     "measured_value_shift": round(float(np.mean(shifts)), 4)})

    return pd.DataFrame(rows)


def audit_augmentations(images: pd.DataFrame, n_per_group: int = 100,
                        n_jitter: int = 40, seed: int = 0) -> dict:
    """Run the full A3.5(a) audit and flag every setting that exceeds the class gap."""
    summary, gap = class_color_gap(images, n_per_group=n_per_group, seed=seed)
    shifts = jitter_shift(images, n_images=n_jitter, seed=seed)

    shifts["class_gap"] = np.where(shifts.parameter == "hue",
                                   gap["hue_gap_deg"], gap["value_gap"])
    measured = np.where(shifts.parameter == "hue",
                        shifts.measured_hue_shift_deg, shifts.measured_value_shift)
    shifts["measured"] = measured
    shifts["ratio_to_class_gap"] = (shifts.measured / shifts.class_gap).round(2)
    shifts["exceeds_class_gap"] = shifts.measured > shifts.class_gap

    return {"class_summary": summary, "gap": gap, "shifts": shifts}
