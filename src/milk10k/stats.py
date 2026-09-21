"""Metadata <-> target association analysis (Part 1).

Unit of analysis
----------------
Every lesion has exactly two images and every patient/lesion field (age,
sex, site, skin tone, ...) is identical on both. Testing those fields on the
10,480-row *image* table would count each lesion twice and halve every
p-value for free, so lesion-level fields are tested on the 5,240-row
*lesion* table. Only the fields that genuinely differ between the two images
of a lesion (``image_type``, ``image_manipulation``, the ``MONET_*`` scores)
are tested at image level.

Statistics
----------
* categorical field vs categorical target -> Pearson chi-square test of
  independence on the contingency table, plus **Cramér's V** (bias-corrected,
  Bergsma 2013) as an effect size in 0-1 so fields with different numbers of
  categories can be compared. p-values alone are useless here: with n = 5,240
  almost everything is "significant".
* numeric / ordinal field vs categorical target -> **Kruskal-Wallis H**
  (the rank-based one-way ANOVA), plus **epsilon-squared** as the effect
  size (share of rank variance explained by class). Chosen over classic
  ANOVA because ``age_approx`` is 5-year-binned and skewed, the MONET scores
  are bounded 0-1 and far from normal, and class sizes range from 9 to 2,522
  lesions - all conditions under which the F-test's assumptions fail while
  the rank test stays valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .data import CLASS_CODES

# --- which fields live where ---------------------------------------------------
ID_FIELDS = ("isic_id", "lesion_id", "path")
TARGET_FIELDS = ("label", "label_name", "malignant")
# derived from / equivalent to the diagnosis -> never a legitimate feature
LEAKAGE_FIELDS = {
    "diagnosis_1": "the target's own benign/malignant level",
    "diagnosis_2": "the target's own second hierarchy level",
    "diagnosis_3": "the target's own third hierarchy level",
    "diagnosis_4": "the target's own fourth hierarchy level",
    "diagnosis_full": "the 48-class fine diagnosis the 11-class label is rolled up from",
    "melanocytic": "True only for melanocytic diagnoses (NV/MEL), NaN otherwise - a label proxy",
    "invasion_thickness_interval": "Breslow thickness; recorded only for invasive melanoma",
}
BIAS_FIELDS = {
    "diagnosis_confirm_type": "biopsied vs clinically assessed - benign lesions are the non-biopsied ones",
    "concomitant_biopsy": "whether a biopsy was taken - a decision made *because of* the suspected diagnosis",
    "image_manipulation": "post-processing flag - acquisition artefact, not lesion biology",
    "skin_tone_class": "legitimate prior, but also a proxy for which country/centre contributed the lesion",
}
# plain explanations for fields that are easy to misread
FIELD_NOTES = {
    "site": "fine-grained anatomical site (8 levels, complete) - NOT the collection centre",
    "anatom_site_general": "coarse anatomical site; the 37% 'missing' rows are exactly the trunk lesions",
    "anatom_site_special": "acral / oral-genital flag, set for only 2% of lesions",
}
IMAGE_LEVEL_FIELDS = ("image_type", "image_manipulation")


@dataclass
class FieldDescription:
    field: str
    kind: str
    level: str
    n_unique: int
    pct_missing: float
    note: str


def describe_fields(images: pd.DataFrame) -> pd.DataFrame:
    """One row per non-id, non-target column: type, level, missingness, caveat."""
    rows: list[FieldDescription] = []
    for col in images.columns:
        if col in ID_FIELDS or col in TARGET_FIELDS:
            continue
        s = images[col]
        n_unique = int(s.nunique(dropna=True))
        non_null = s.dropna()
        if n_unique <= 1:
            kind = "flag (True/NaN)" if s.isna().any() else "constant"
        elif s.dtype == bool or set(non_null.unique()) <= {True, False}:
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "ordinal" if col == "skin_tone_class" else "numeric"
        elif n_unique > 0.5 * len(non_null):        # (almost) every value distinct
            kind = "free text"
        else:
            kind = "categorical"
        level = "image" if (col in IMAGE_LEVEL_FIELDS or col.startswith("MONET_")) else "lesion"
        note = LEAKAGE_FIELDS.get(col) and f"LEAKAGE: {LEAKAGE_FIELDS[col]}" \
            or BIAS_FIELDS.get(col) and f"BIAS RISK: {BIAS_FIELDS[col]}" \
            or FIELD_NOTES.get(col) \
            or ("derived from the image by a vision-language model, not clinical metadata"
                if col.startswith("MONET_") else "")
        rows.append(FieldDescription(col, kind, level, n_unique,
                                     round(float(s.isna().mean() * 100), 1), note))
    return pd.DataFrame([r.__dict__ for r in rows])


# --- statistics ----------------------------------------------------------------
def cramers_v(table: pd.DataFrame | np.ndarray) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013) for a contingency table."""
    obs = np.asarray(table, dtype=float)
    n = obs.sum()
    if n == 0:
        return float("nan")
    chi2 = sps.chi2_contingency(obs, correction=False)[0]
    phi2 = chi2 / n
    r, k = obs.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else float("nan")


def categorical_vs_target(df: pd.DataFrame, field: str, target: str = "label",
                          min_count: int = 1) -> dict:
    """Chi-square test + Cramér's V for one categorical field.

    Rows with a missing field value are dropped (missingness is reported
    separately). Categories with fewer than ``min_count`` rows are merged
    into ``"(rare, merged)"`` so the chi-square expected counts stay sane.
    """
    sub = df[[field, target]].dropna()
    col = sub[field].astype(str)
    small = col.value_counts()[lambda s: s < min_count].index
    col = col.where(~col.isin(small), "(rare, merged)")
    ct = pd.crosstab(col, sub[target])
    ct = ct.reindex(columns=[c for c in CLASS_CODES if c in ct.columns])
    chi2, p, dof, _ = sps.chi2_contingency(ct.values, correction=False)
    return {
        "field": field, "test": "chi-square", "n": int(len(sub)),
        "statistic": float(chi2), "dof": int(dof), "p_value": float(p),
        "effect_size": cramers_v(ct), "effect_name": "Cramér's V",
        "n_categories": int(ct.shape[0]),
        "counts": ct,
        "class_share": ct.div(ct.sum(axis=1), axis=0),   # P(class | category)
    }


def epsilon_squared(h: float, n: int, k: int) -> float:
    """Effect size for Kruskal-Wallis: (H - k + 1) / (n - k), clipped at 0."""
    return float(max(0.0, (h - k + 1) / (n - k))) if n > k else float("nan")


def numeric_vs_target(df: pd.DataFrame, field: str, target: str = "label") -> dict:
    """Kruskal-Wallis H + epsilon-squared for one numeric/ordinal field."""
    sub = df[[field, target]].dropna()
    groups = [g[field].to_numpy(dtype=float) for _, g in sub.groupby(target)]
    groups = [g for g in groups if len(g) > 0]
    h, p = sps.kruskal(*groups)
    per_class = (sub.groupby(target)[field]
                 .agg(n="size", mean="mean", median="median", std="std")
                 .reindex([c for c in CLASS_CODES if c in set(sub[target])]))
    return {
        "field": field, "test": "Kruskal-Wallis", "n": int(len(sub)),
        "statistic": float(h), "dof": int(len(groups) - 1), "p_value": float(p),
        "effect_size": epsilon_squared(h, len(sub), len(groups)), "effect_name": "epsilon²",
        "n_categories": int(len(groups)),
        "per_class": per_class,
    }


def association_table(lesions: pd.DataFrame, images: pd.DataFrame,
                      target: str = "label") -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run the right test for every usable field; return (summary, per-field results).

    ``summary`` has one row per field with the test used, n, statistic,
    p-value, effect size and a caveat column, sorted by ``effect_r`` -
    Cramér's V for categorical fields and sqrt(epsilon²) for numeric ones, so
    the two families are ranked on the same (correlation-like) scale.
    """
    desc = describe_fields(images)
    results: dict[str, dict] = {}
    for _, d in desc.iterrows():
        if d.kind in ("constant", "flag (True/NaN)") or d.field in LEAKAGE_FIELDS:
            continue
        table = images if d.level == "image" else lesions
        if d.field not in table.columns:
            continue
        if d.kind in ("numeric", "ordinal"):
            res = numeric_vs_target(table, d.field, target)
        else:
            res = categorical_vs_target(table, d.field, target, min_count=10)
        res.update(kind=d.kind, level=d.level, pct_missing=d.pct_missing, note=d.note)
        results[d.field] = res
    cols = ["field", "kind", "level", "pct_missing", "test", "n", "statistic",
            "dof", "p_value", "effect_name", "effect_size", "note"]
    summary = pd.DataFrame([{k: r[k] for k in cols} for r in results.values()])
    # Cramér's V is on a correlation scale, epsilon² on a variance-explained
    # (squared) scale; sqrt(epsilon²) puts the two on a common footing for ranking.
    summary["effect_r"] = np.where(summary["effect_name"] == "epsilon²",
                                   np.sqrt(summary["effect_size"]), summary["effect_size"])
    summary = summary.sort_values("effect_r", ascending=False).reset_index(drop=True)
    return summary, results


def lesion_table_for_stats(images: pd.DataFrame) -> pd.DataFrame:
    """One row per lesion with *every* lesion-level field (first value)."""
    lesion_cols = [c for c in images.columns
                   if c not in ID_FIELDS and c not in IMAGE_LEVEL_FIELDS
                   and not c.startswith("MONET_")]
    return (images.sort_values("image_type")
            .groupby("lesion_id", as_index=False)[lesion_cols].first())
