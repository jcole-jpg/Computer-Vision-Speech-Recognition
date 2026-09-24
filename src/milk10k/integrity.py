"""Data-trust checks: label-file cross-checks (A1.1) and full-dataset integrity (B1).

Everything here returns a *result object* rather than printing, so the same code
backs the notebook (which prints it) and the tests (which assert on it). A check
that passes is boring; a check that fails must say exactly which rows broke it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image

from .config import IMAGE_DIR
from .data import CLASS_CODES, load_metadata, load_training_gt

# --------------------------------------------------------------------------
# A1.1 - cross-check the two label files
# --------------------------------------------------------------------------


@dataclass
class CheckResult:
    """One named check: did it pass, and what exactly violated it?"""

    name: str
    passed: bool
    detail: str = ""
    violations: pd.DataFrame | None = None

    @property
    def n_violations(self) -> int:
        return 0 if self.violations is None else len(self.violations)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        extra = f"  ({self.n_violations} violations)" if not self.passed else ""
        return f"[{mark}] {self.name}: {self.detail}{extra}"


def check_lesion_ids_match(metadata: pd.DataFrame, gt: pd.DataFrame) -> CheckResult:
    """(a) The set of lesion_id is identical in metadata.csv and training_gt.csv."""
    meta_ids, gt_ids = set(metadata["lesion_id"]), set(gt["lesion_id"])
    only_meta, only_gt = meta_ids - gt_ids, gt_ids - meta_ids
    passed = not only_meta and not only_gt
    violations = None
    if not passed:
        violations = pd.DataFrame(
            [{"lesion_id": i, "only_in": "metadata"} for i in sorted(only_meta)]
            + [{"lesion_id": i, "only_in": "training_gt"} for i in sorted(only_gt)]
        )
    return CheckResult(
        "a. lesion_id sets identical",
        passed,
        f"{len(meta_ids)} in metadata, {len(gt_ids)} in gt, "
        f"{len(only_meta)} metadata-only, {len(only_gt)} gt-only",
        violations,
    )


def check_one_hot(gt: pd.DataFrame) -> CheckResult:
    """(b) Every lesion has exactly ONE positive class in the one-hot ground truth."""
    row_sum = gt[CLASS_CODES].sum(axis=1)
    bad = row_sum != 1
    violations = None
    if bad.any():
        violations = gt.loc[bad, ["lesion_id"]].assign(n_positive=row_sum[bad])
    return CheckResult(
        "b. exactly one positive class per lesion",
        not bad.any(),
        f"{(~bad).sum()}/{len(gt)} lesions have exactly one positive class",
        violations,
    )


def check_class_to_diagnosis1(lesions: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame]:
    """(c) Does each of the 11 classes map to exactly ONE diagnosis_1 value?

    Returns the check *and* the full class -> diagnosis_1 table with counts,
    because that table is the answer to the written question, not a by-product.
    """
    table = (
        lesions.groupby(["dx", "diagnosis_1"]).size()
        .rename("n_lesions").reset_index()
        .sort_values(["dx", "n_lesions"], ascending=[True, False])
    )
    per_class = table.groupby("dx")["diagnosis_1"].nunique()
    ambiguous = per_class[per_class > 1].index.tolist()
    violations = table[table["dx"].isin(ambiguous)] if ambiguous else None
    return (
        CheckResult(
            "c. each class maps to exactly one diagnosis_1",
            not ambiguous,
            f"{len(per_class) - len(ambiguous)}/{len(per_class)} classes are unambiguous"
            + (f"; ambiguous: {', '.join(ambiguous)}" if ambiguous else ""),
            violations,
        ),
        table,
    )


def check_hierarchy(metadata: pd.DataFrame) -> list[CheckResult]:
    """(d) diagnosis_3 -> diagnosis_2 -> diagnosis_1 is a strict tree.

    Each child value must belong to exactly one parent value. NaNs are dropped:
    an unrecorded level is missing data, not a hierarchy violation.
    """
    results = []
    for child, parent in [("diagnosis_3", "diagnosis_2"), ("diagnosis_2", "diagnosis_1")]:
        sub = metadata[[child, parent]].dropna()
        n_parents = sub.groupby(child)[parent].nunique()
        bad = n_parents[n_parents > 1]
        violations = None
        if len(bad):
            violations = (
                sub[sub[child].isin(bad.index)]
                .drop_duplicates().sort_values(child).reset_index(drop=True)
            )
        results.append(
            CheckResult(
                f"d. each {child} has exactly one {parent}",
                not len(bad),
                f"{len(n_parents)} distinct {child} values, {len(bad)} with >1 parent",
                violations,
            )
        )
    return results


#: Fields that describe the *lesion*, so both of its images must agree (A1.1e).
LESION_LEVEL_FIELDS = (
    "age_approx", "sex", "anatom_site_general",
    "diagnosis_1", "diagnosis_2", "diagnosis_3",
)


def check_paired_images(metadata: pd.DataFrame,
                        fields: tuple[str, ...] = LESION_LEVEL_FIELDS) -> CheckResult:
    """(e) A lesion's two images carry identical age, sex, site and diagnosis fields."""
    rows = []
    for f in fields:
        if f not in metadata.columns:
            continue
        # dropna=False so that "one image has a site, the other is NaN" counts.
        n_unique = metadata.groupby("lesion_id")[f].nunique(dropna=False)
        bad = n_unique[n_unique > 1]
        rows.append({"field": f, "n_inconsistent_lesions": int(len(bad))})
    summary = pd.DataFrame(rows)
    total = int(summary["n_inconsistent_lesions"].sum())
    return CheckResult(
        "e. paired images agree on lesion-level fields",
        total == 0,
        f"{total} field/lesion inconsistencies across {len(summary)} fields",
        summary if total else None,
    )


def run_label_checks(metadata: pd.DataFrame | None = None,
                     gt: pd.DataFrame | None = None,
                     lesions: pd.DataFrame | None = None
                     ) -> tuple[list[CheckResult], pd.DataFrame]:
    """Run A1.1 (a)-(e) and return (results, class->diagnosis_1 table)."""
    from .data import build_lesion_table

    metadata = load_metadata() if metadata is None else metadata
    gt = load_training_gt() if gt is None else gt
    lesions = build_lesion_table() if lesions is None else lesions

    class_check, class_table = check_class_to_diagnosis1(lesions)
    results = [
        check_lesion_ids_match(metadata, gt),
        check_one_hot(gt),
        class_check,
        *check_hierarchy(metadata),
        check_paired_images(metadata),
    ]
    return results, class_table


# --------------------------------------------------------------------------
# B1 - full-dataset integrity
# --------------------------------------------------------------------------


@dataclass
class IntegrityReport:
    """Result of the full-image-set scan (B1)."""

    n_rows: int
    missing_files: list[str] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)
    sizes: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ok(self) -> bool:
        return not self.missing_files and not self.unreadable_files

    def size_summary(self) -> pd.DataFrame:
        """min / median / max width and height, overall and per image_type."""
        if self.sizes.empty:
            return pd.DataFrame()

        def agg(g: pd.DataFrame, scope: str) -> dict:
            return {
                "scope": scope, "n_images": len(g),
                "width_min": int(g.width.min()), "width_median": float(g.width.median()),
                "width_max": int(g.width.max()),
                "height_min": int(g.height.min()), "height_median": float(g.height.median()),
                "height_max": int(g.height.max()),
                "bytes_min": int(g.n_bytes.min()), "bytes_median": float(g.n_bytes.median()),
                "bytes_max": int(g.n_bytes.max()),
            }

        rows = [agg(self.sizes, "all")]
        if "image_type" in self.sizes.columns:
            rows += [agg(g, t) for t, g in self.sizes.groupby("image_type")]
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        return (
            f"rows checked      : {self.n_rows}\n"
            f"missing files     : {len(self.missing_files)}\n"
            f"unreadable files  : {len(self.unreadable_files)}\n"
            f"integrity         : {'OK' if self.ok else 'PROBLEMS FOUND'}"
        )


def verify_images(images: pd.DataFrame, image_dir: Path | None = None,
                  *, verify: bool = True) -> IntegrityReport:
    """Check that every row resolves to a readable image, and record its geometry.

    ``Image.verify()`` parses the file's structure without decoding the pixels,
    so it catches truncation/corruption cheaply. It invalidates the handle, so
    the file is reopened to read ``size`` - that is why each file is opened twice.
    """
    image_dir = Path(image_dir) if image_dir is not None else IMAGE_DIR
    missing: list[str] = []
    unreadable: list[str] = []
    rows: list[dict] = []

    paths = (images["path"] if "path" in images.columns
             else images["isic_id"].map(lambda i: image_dir / f"{i}.jpg"))

    for isic_id, path, itype in zip(images["isic_id"], paths,
                                    images.get("image_type", pd.Series(["?"] * len(images)))):
        path = Path(path)
        if not path.is_file():
            missing.append(isic_id)
            continue
        try:
            if verify:
                with Image.open(path) as im:
                    im.verify()
            with Image.open(path) as im:
                w, h = im.size
                mode = im.mode
        except Exception:  # noqa: BLE001 - any decode error means "unreadable"
            unreadable.append(isic_id)
            continue
        rows.append({"isic_id": isic_id, "image_type": itype, "width": w,
                     "height": h, "mode": mode, "n_bytes": path.stat().st_size})

    return IntegrityReport(len(images), missing, unreadable, pd.DataFrame(rows))
