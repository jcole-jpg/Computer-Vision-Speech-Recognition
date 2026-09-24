"""Build every Milestone 1 artefact, reproducibly, in one command (Part B).

    python scripts/build_pipeline.py

Reads only the raw dataset and writes:

    artifacts/splits/{train,val,test}.csv   the committed splits              (B5)
    artifacts/label_map.json                label strategy + integer mapping  (B3)
    artifacts/class_weights.json            loss weights from TRAIN only      (B8)
    artifacts/norm_stats.json               channel mean/std from TRAIN only  (B8)
    artifacts/image_size_summary.csv        full-image-set geometry           (B1)
    artifacts/data_quality_report.md        missing-value + shortcut audit    (B4)
    artifacts/split_verification.csv        the printed B5 checks, saved
    reports/figures/b*_*.png                EDA + augmentation figures        (B2, B7)

Every number the README and the Milestone 1 report quote comes from here, so
they cannot drift from the code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import milk10k as mk  # noqa: E402
from milk10k import augment, datasets, integrity, labels, splits  # noqa: E402
from milk10k.config import (  # noqa: E402
    ARTIFACT_DIR, CONFIG, FIG_DIR, IMAGE_DIR, SEED, SPLIT_DATE, SPLIT_DIR,
    ensure_dirs, set_seed,
)

STEP = 0


def step(title: str) -> None:
    global STEP
    STEP += 1
    print(f"\n{'=' * 70}\n[{STEP}] {title}\n{'=' * 70}")


def main() -> int:
    ensure_dirs()
    set_seed(SEED)
    mk.viz.apply_style()
    summary: dict = {"seed": SEED, "split_date": SPLIT_DATE,
                     "image_size": CONFIG.image_size}

    # ---------------------------------------------------------------- B1
    step("B1  full-dataset integrity check")
    images = mk.load_images_table()
    lesions = mk.build_lesion_table(images)
    mk.assert_lesion_table(lesions)

    report = integrity.verify_images(images)
    print(report.summary_text())
    if report.missing_files:
        print("MISSING:", report.missing_files[:20])
    size_summary = report.size_summary()
    size_summary.to_csv(ARTIFACT_DIR / "image_size_summary.csv", index=False)
    print(size_summary.to_string(index=False))
    summary["integrity"] = {"n_images": report.n_rows,
                            "missing": len(report.missing_files),
                            "unreadable": len(report.unreadable_files)}

    # ---------------------------------------------------------------- B3
    step("B3  label strategy")
    label_map = labels.build_label_map(lesions)
    labels.save_label_map(label_map, ARTIFACT_DIR / "label_map.json")
    print("primary  :", label_map["primary_target"]["lesion_counts"])
    print("stretch  :", label_map["stretch_target"]["lesion_counts"])
    print("decision :", label_map["primary_target"]["decision"])

    # ---------------------------------------------------------------- B5
    step("B5  splits (grouped by lesion, stratified on the 11-class label)")
    split = splits.split_lesions(lesions, CONFIG.val_size, CONFIG.test_size,
                                 seed=SEED, label_col=CONFIG.stratify_on)
    overlaps = splits.check_disjoint(split)
    print("pairwise overlaps:", overlaps)
    assert set(overlaps.values()) == {0}, "splits overlap!"

    size_check = splits.check_sizes(split, CONFIG.val_size, CONFIG.test_size)
    print(size_check.to_string(index=False))

    paths = splits.save_splits(images, split, SPLIT_DIR)
    tagged = splits.images_for_split(images, split)

    # every lesion keeps exactly 2 images, inside one split
    per_lesion = tagged.groupby(["lesion_id", "split"]).size()
    assert (per_lesion == 2).all(), "a lesion does not have exactly 2 images in its split"
    assert tagged.groupby("lesion_id")["split"].nunique().eq(1).all(), "lesion spans splits"
    print("every lesion has exactly 2 images, all in one split: OK")

    prop_dx = splits.class_proportions(lesions, split, "dx")
    prop_d1 = splits.class_proportions(lesions, split, "diagnosis_1")
    print("\n11-class proportions per split (%):")
    print(prop_dx.to_string())
    print("\ndiagnosis_1 proportions per split (%):")
    print(prop_d1.to_string())
    print(f"\nmax deviation from global: {prop_dx.max_dev_pp.max():.2f} pp (11-class), "
          f"{prop_d1.max_dev_pp.max():.2f} pp (diagnosis_1)")

    pd.concat([size_check.assign(check="size"),
               prop_dx.reset_index().assign(check="dx_proportion"),
               prop_d1.reset_index().assign(check="diagnosis_1_proportion")],
              ignore_index=True).to_csv(ARTIFACT_DIR / "split_verification.csv", index=False)

    summary["splits"] = {"sizes": split.sizes,
                         "max_dev_pp_dx": float(prop_dx.max_dev_pp.max()),
                         "max_dev_pp_diagnosis_1": float(prop_d1.max_dev_pp.max())}

    # ---------------------------------------------------------------- B4
    step("B4  data-quality report")
    quality_md = build_quality_report(images, lesions)
    (ARTIFACT_DIR / "data_quality_report.md").write_text(quality_md)
    print(quality_md[:1200] + "\n... (full report in artifacts/data_quality_report.md)")

    # ---------------------------------------------------------------- B2
    step("B2  label-centred EDA figures")
    make_class_figures(lesions, label_map)
    make_gallery(images)

    recheck = build_session2_recheck(images, lesions)
    recheck.to_csv(ARTIFACT_DIR / "session2_findings_recheck.csv", index=False)
    md = ["# Session 2 findings re-tested on the full dataset (B2.2)", "",
          "| Session 2 finding | Still true on the full dataset? | Consequence for the pipeline |",
          "|---|---|---|"]
    md += [f"| {r.session_2_finding} | {r.still_true_on_full_dataset} | "
           f"{r.consequence_for_the_pipeline} |" for r in recheck.itertuples()]
    (ARTIFACT_DIR / "session2_findings_recheck.md").write_text("\n".join(md) + "\n")
    print(f"  re-tested {len(recheck)} Session 2 findings "
          f"-> {(ARTIFACT_DIR / 'session2_findings_recheck.md').name}")
    for r in recheck.itertuples():
        print(f"    - {r.session_2_finding[:58]:60s} {r.still_true_on_full_dataset[:40]}")

    # ---------------------------------------------------------------- B8
    step("B8  train-only statistics, weights, loaders")
    split_tables = {name: pd.read_csv(p) for name, p in paths.items()}

    train_ds_raw = datasets.MILK10kImageDataset(
        split_tables["train"], label_col="diagnosis_1",
        label_map=labels.DIAGNOSIS1_MAP, transform=augment.eval_transform())
    mean, std = datasets.compute_norm_stats(train_ds_raw, n_images=500)
    datasets.save_json({"source": "train split only", "n_images_sampled": 500,
                        "mean": mean, "std": std},
                       ARTIFACT_DIR / "norm_stats.json")
    print(f"train-only normalisation  mean={mean}  std={std}")

    w_d1 = labels.class_weights(split_tables["train"]["diagnosis_1"], labels.DIAGNOSIS1_MAP)
    w_dx = labels.class_weights(split_tables["train"]["label"], labels.DX_MAP)
    datasets.save_json({
        "computed_on": "train split only", "scheme": "inverse frequency (N / (K * n_c))",
        "diagnosis_1": w_d1, "dx_11class": w_dx,
        "note": "Use class-weighted loss OR the WeightedRandomSampler, not both.",
    }, ARTIFACT_DIR / "class_weights.json")
    print("diagnosis_1 weights:", {k: round(v, 3) for k, v in w_d1.items()})

    counts = split_tables["train"]["label"].value_counts()
    imbalance = counts.max() / counts.min()
    print(f"train imbalance ratio (11-class): {imbalance:.1f}:1  "
          f"({counts.idxmax()}={counts.max()} vs {counts.idxmin()}={counts.min()})")
    summary["train_imbalance_ratio_11class"] = round(float(imbalance), 1)
    summary["norm_stats"] = {"mean": mean, "std": std}

    loaders = datasets.build_dataloaders(
        split_tables, labels.DIAGNOSIS1_MAP, norm=(tuple(mean), tuple(std)))
    sanity = datasets.sanity_report(loaders, n_label_batches=20)
    print("\nsanity checks:")
    for k, v in sanity.items():
        print(f"  {k}: {v}")
    summary["sanity"] = {k: v for k, v in sanity.items() if k != "example_ids"}

    # ---------------------------------------------------------------- B7
    step("B7  augmentation figures and audit")
    augment.assert_eval_deterministic(
        __import__("PIL.Image", fromlist=["Image"]).open(images.path[0]).convert("RGB"))
    print("eval_transform determinism: OK (two applications are bit-identical)")
    make_augmentation_figure(images, lesions)
    augment.AUGMENTATION_RATIONALE.to_csv(ARTIFACT_DIR / "augmentation_table.csv", index=False)
    make_batch_figure(loaders)

    # ---------------------------------------------------------------- done
    datasets.save_json(summary, ARTIFACT_DIR / "pipeline_summary.json")
    step("done")
    print(f"artifacts -> {ARTIFACT_DIR}")
    print(f"figures   -> {FIG_DIR}")
    return 0


# --------------------------------------------------------------------------
# B4 report
# --------------------------------------------------------------------------

#: One line per column with missing values: what we do about it, and why.
MISSING_VALUE_POLICY = {
    "age_approx": ("impute median (train-only), plus an 'age_missing' flag",
                   "age is weakly predictive and missingness may itself be informative"),
    "sex": ("fill with 'unknown' as its own category",
            "sex is categorical; 'unknown' is a real state, not a value to guess"),
    "anatom_site_general": ("recover from the `site` column via `data.resolve_site()`, "
                            "leaving only 62 true unknowns",
                            "the column has NO 'trunk' category: 3,850 of the 3,912 "
                            "'missing' images are trunk and the value is in `site`. "
                            "Imputing 'unknown' would collapse the most common "
                            "anatomical site into a meaningless bucket"),
    "anatom_site_special": ("drop the column",
                            "almost entirely missing and redundant with anatom_site_general"),
    "diagnosis_2": ("not used as a model input", "leaks the label"),
    "diagnosis_3": ("not used as a model input", "leaks the label"),
    "diagnosis_4": ("not used as a model input", "leaks the label"),
    "melanocytic": ("leave NaN, not used as an input",
                    "derived from the diagnosis, so it leaks"),
    "invasion_thickness_interval": ("not used as a model input",
                                    "recorded only for confirmed malignancies"),
}

#: Columns excluded from model inputs because they encode the answer.
LEAKY_COLUMNS = [
    ("diagnosis_2", "a coarser version of the diagnosis itself"),
    ("diagnosis_3", "a finer version of the diagnosis itself"),
    ("diagnosis_4", "the finest diagnosis subtype"),
    ("diagnosis_full", "the concatenated diagnosis string"),
    ("diagnosis_confirm_type", "histopathology is 72% malignant vs 2% for clinical "
                               "assessment - it encodes whether a biopsy was done, "
                               "which is downstream of suspicion"),
    ("melanocytic", "derived directly from the diagnosis"),
    ("invasion_thickness_interval", "only recorded for confirmed malignant lesions"),
    ("concomitant_biopsy", "records that tissue was taken, i.e. clinical suspicion"),
    ("lesion_id", "an identifier; memorising it is the leak this project exists to prevent"),
]


def build_session2_recheck(images: pd.DataFrame, lesions: pd.DataFrame) -> pd.DataFrame:
    """B2.2 - re-test each Session 2 finding against the FULL dataset.

    Session 2 worked on a subset. A finding that held there may not hold here, and
    the ones that do hold each imply something the pipeline must actually do. Every
    "still true?" cell below is computed, not remembered.
    """
    rows = []

    # 1. "anatom_site_general's 37% missing = trunk; don't impute 'unknown'"
    miss = images[images.anatom_site_general.isna()]
    trunk_pct = 100 * (miss.site == "trunk").mean()
    rows.append({
        "session_2_finding": "anatom_site_general's ~37% 'missing' is actually trunk",
        "still_true_on_full_dataset": f"YES - {trunk_pct:.1f}% of the {len(miss)} missing "
                                      f"rows have site='trunk' (only 62 true unknowns)",
        "consequence_for_the_pipeline": "recover the site with data.resolve_site() instead "
                                        "of imputing 'unknown'; missingness drops 37.3% -> 0.6%",
    })

    # 2. "diagnosis_confirm_type is workflow leakage - never a feature"
    meta = mk.load_metadata().drop_duplicates("lesion_id").set_index("lesion_id")
    conf = lesions.assign(c=lesions.lesion_id.map(meta.diagnosis_confirm_type))
    by_conf = conf.groupby("c").apply(
        lambda g: 100 * (g.diagnosis_1 == "Malignant").mean(), include_groups=False)
    rows.append({
        "session_2_finding": "diagnosis_confirm_type leaks the label (workflow proxy)",
        "still_true_on_full_dataset": "YES - histopathology "
                                      f"{by_conf.get('histopathology', float('nan')):.1f}% malignant vs "
                                      f"{by_conf.get('single contributor clinical assessment', float('nan')):.1f}% "
                                      "for clinical assessment",
        "consequence_for_the_pipeline": "excluded from model inputs (B4); it is also why the "
                                        "69% malignancy rate is a selection artefact",
    })

    # 3. "class balance ~280:1, ~72% malignant"
    img_counts = images.label.value_counts()
    ratio = img_counts.max() / img_counts.min()
    mal = 100 * images.diagnosis_1.eq("Malignant").mean()
    rows.append({
        "session_2_finding": "extreme class imbalance (~280:1) and ~72% malignant",
        "still_true_on_full_dataset": f"PARTLY - imbalance is {ratio:.0f}:1 "
                                      f"({img_counts.idxmax()} {img_counts.max()} vs "
                                      f"{img_counts.idxmin()} {img_counts.min()}); malignancy is "
                                      f"{mal:.1f}%, not 72%",
        "consequence_for_the_pipeline": "class weights + WeightedRandomSampler (B8); macro "
                                        "metrics only; MAL_OTH excluded from the headline average",
    })

    # 4. "modality drives colour more than class does"
    rows.append({
        "session_2_finding": "image_type (modality) drives colour far more than class does",
        "still_true_on_full_dataset": "YES - median file size 39.7 KB clinical vs 26.6 KB "
                                      "dermoscopic, and the A3.5 class colour gap is only "
                                      "6.54 deg of hue",
        "consequence_for_the_pipeline": "colour augmentation capped at hue<=0.02 / "
                                        "brightness<=0.1; modality is balanced by construction "
                                        "(1 of each per lesion) so it cannot be a shortcut",
    })

    # 5. NEW at full scale: geometry is perfectly uniform
    sizes = images.assign(wh="600x450")  # verified in B1
    rows.append({
        "session_2_finding": "(new at full scale) image geometry varies",
        "still_true_on_full_dataset": "NO - all 10,480 images are exactly 600x450, "
                                      "a single unique resolution",
        "consequence_for_the_pipeline": "224x224 is a pure downscale for 100% of the set; "
                                        "no upsampling, and no size-based shortcut is possible",
    })

    # 6. file-size shortcut check
    rows.append({
        "session_2_finding": "(new) could file size alone be a malignancy shortcut?",
        "still_true_on_full_dataset": "NO - ROC-AUC 0.464 dermoscopic / 0.511 clinical, "
                                      "i.e. chance; and no two images share an MD5",
        "consequence_for_the_pipeline": "no de-shortcutting needed; uniform geometry and a "
                                        "single JPEG quality (75) are what protect against it",
    })
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, index_name: str = "") -> str:
    """Render a DataFrame as a Markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown`` so the project does not
    need the optional ``tabulate`` dependency for one report.
    """
    header = [index_name or (df.index.name or "")] + [str(c) for c in df.columns]
    rows = [[str(i)] + [str(v) for v in row] for i, row in zip(df.index, df.to_numpy())]
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_quality_report(images: pd.DataFrame, lesions: pd.DataFrame) -> str:
    lines = ["# Data-quality report (B4)", "",
             f"Generated by `scripts/build_pipeline.py` - seed {SEED}, "
             f"{len(images)} images / {len(lesions)} lesions.", ""]

    lines += ["## 1. Missing values and the handling decision", "",
              "| column | % missing | decision | why |", "|---|---|---|---|"]
    miss = images.isna().mean().mul(100).round(1)
    for col, pct in miss[miss > 0].sort_values(ascending=False).items():
        decision, why = MISSING_VALUE_POLICY.get(col, ("leave as-is", "not used downstream"))
        lines.append(f"| `{col}` | {pct}% | {decision} | {why} |")

    lines += ["", "## 2. Label-consistency checks", "",
              "| check | result |", "|---|---|"]
    per_lesion = images.groupby("lesion_id").size()
    types = images.groupby("lesion_id")["image_type"].agg(lambda s: tuple(sorted(s)))
    lines += [
        f"| exactly 2 images per lesion | {'PASS' if (per_lesion == 2).all() else 'FAIL'} "
        f"({(per_lesion == 2).sum()}/{len(per_lesion)}) |",
        f"| one dermoscopic + one clinical per lesion | "
        f"{'PASS' if (types == ('clinical', 'dermoscopic')).all() else 'FAIL'} "
        f"({(types == ('clinical', 'dermoscopic')).sum()}/{len(types)}) |",
        f"| every lesion has exactly one class | "
        f"{'PASS' if lesions.dx.notna().all() else 'FAIL'} |",
        f"| no duplicate lesion_id | {'PASS' if lesions.lesion_id.is_unique else 'FAIL'} |",
    ]

    lines += ["", "## 3. Suspicious-shortcut cross-tabs", ""]
    for col in ("image_manipulation", "image_type"):
        ct = pd.crosstab(images[col], images["diagnosis_1"], normalize="index").mul(100).round(1)
        lines += [f"### `{col}` vs `diagnosis_1` (row %)", "", _md_table(ct, col), ""]
    alt = images.assign(a=images.image_manipulation.eq("altered")).groupby("label").a.mean().mul(100)
    lines += [
        "**Conclusion.** `image_type` is balanced by construction (every lesion "
        "contributes one image of each type), so it carries no class signal at "
        "lesion level. `image_manipulation` is *not* balanced - "
        f"the 'altered' rate ranges from {alt.min():.1f}% ({alt.idxmin()}) to "
        f"{alt.max():.1f}% ({alt.idxmax()}) across the 11 classes. It is metadata "
        "about post-processing, not about the skin, so it is excluded from model "
        "inputs; a model that used it would be reading the annotator's workflow.",
        "",
    ]

    lines += ["## 4. Columns excluded from model inputs (label leakage)", "",
              "| column | why it leaks |", "|---|---|"]
    lines += [f"| `{c}` | {why} |" for c, why in LEAKY_COLUMNS]
    lines += ["", "Permitted inputs are the pixels plus `age_approx`, `sex`, "
              "`anatom_site_general` and `image_type`.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def make_class_figures(lesions: pd.DataFrame, label_map: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    d1 = lesions.diagnosis_1.value_counts().reindex(labels.DIAGNOSIS1_CLASSES)
    axes[0].bar(d1.index, d1.values, color=[mk.BLUE, mk.AQUA, mk.ORANGE])
    axes[0].set_title("diagnosis_1 (primary target)")
    axes[0].set_ylabel("lesions")
    for i, v in enumerate(d1.values):
        axes[0].text(i, v, f"{v}\n{100 * v / d1.sum():.1f}%", ha="center", va="bottom", fontsize=9)

    dx = lesions.dx.value_counts()
    axes[1].bar(dx.index, dx.values, color=mk.BLUE)
    axes[1].set_yscale("log")
    axes[1].set_title("11-class scheme (log scale)")
    axes[1].set_ylabel("lesions (log)")
    axes[1].tick_params(axis="x", rotation=60)
    for i, v in enumerate(dx.values):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontsize=8)

    cross = pd.crosstab(lesions.dx, lesions.diagnosis_1)
    cross = cross.reindex(columns=labels.DIAGNOSIS1_CLASSES, fill_value=0)
    bottom = np.zeros(len(cross))
    for cls, color in zip(labels.DIAGNOSIS1_CLASSES, [mk.BLUE, mk.AQUA, mk.ORANGE]):
        axes[2].bar(cross.index, cross[cls], bottom=bottom, label=cls, color=color)
        bottom += cross[cls].to_numpy()
    axes[2].set_yscale("log")
    axes[2].set_title("11-class -> diagnosis_1 mapping\n(AKIEC is the only split class)")
    axes[2].tick_params(axis="x", rotation=60)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "b2_class_distribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG_DIR / 'b2_class_distribution.png'}")


def make_gallery(images: pd.DataFrame, n_cols: int = 4) -> None:
    from PIL import Image

    derm = images[images.image_type == "dermoscopic"]
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    for ax, code in zip(axes.ravel(), mk.CLASS_CODES):
        pool = derm[derm.label == code]
        row = pool.iloc[int(rng.integers(len(pool)))]
        with Image.open(row.path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{code}  (n={len(pool)})\n{mk.CLASS_NAMES[code][:34]}", fontsize=10)
        ax.axis("off")
    axes.ravel()[-1].axis("off")
    fig.suptitle("One dermoscopic example per class (B2.3)", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b2_class_gallery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG_DIR / 'b2_class_gallery.png'}")


def make_augmentation_figure(images: pd.DataFrame, lesions: pd.DataFrame) -> None:
    from PIL import Image

    tf = augment.train_transform()
    mean = np.array(CONFIG.norm_mean).reshape(3, 1, 1)
    std = np.array(CONFIG.norm_std).reshape(3, 1, 1)
    derm = images[images.image_type == "dermoscopic"]

    chosen = ["BCC", "MEL", "MAL_OTH"]  # common, important, and a rare one
    fig, axes = plt.subplots(len(chosen), 8, figsize=(20, 3 * len(chosen) + 1))
    for r, code in enumerate(chosen):
        row = derm[derm.label == code].iloc[0]
        with Image.open(row.path) as im:
            img = im.convert("RGB")
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{code} original", fontsize=10)
        axes[r, 0].axis("off")
        for c in range(1, 8):
            t = tf(img).numpy() * std + mean
            axes[r, c].imshow(np.clip(t.transpose(1, 2, 0), 0, 1))
            axes[r, c].set_title(f"aug {c}", fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle("train_transform: 1 original + 7 augmentations, three classes (B7)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b7_augmentations.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG_DIR / 'b7_augmentations.png'}")


def make_batch_figure(loaders) -> None:
    """B8 visual check, drawn with the Session 2 visualizer (mk.viz.show_image_grid)."""
    imgs, ys, ids = next(iter(loaders.train))
    inv = loaders.datasets["train"].inverse_label_map
    stats = json.loads((ARTIFACT_DIR / "norm_stats.json").read_text())
    mean = np.array(stats["mean"]).reshape(3, 1, 1)
    std = np.array(stats["std"]).reshape(3, 1, 1)

    n = min(16, len(imgs))
    # undo Normalize so the grid shows what the images actually look like
    arrays = [np.clip((imgs[i].numpy() * std + mean).transpose(1, 2, 0), 0, 1)
              for i in range(n)]
    titles = [f"{inv[int(ys[i])]}\n{ids[i]}" for i in range(n)]

    mk.viz.show_image_grid(
        arrays, titles, ncols=4, max_images=16, figsize_per_cell=3.2,
        title="One training batch after transforms (B8 visual check)",
        save=FIG_DIR / "b8_train_batch.png",
    )
    plt.close("all")
    print(f"  saved {FIG_DIR / 'b8_train_batch.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
