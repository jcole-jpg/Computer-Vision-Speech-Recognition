# Submission — Homework Sessions 1–3 + Course Project Milestone 1

**Name:** Joshua Cole
**Repository:** https://github.com/jcole-jpg/Computer-Vision-Speech-Recognition
**Course:** Computer Vision and Speech Recognition — EADA
**Dataset:** MILK10k (5,240 lesions / 10,480 images) · seed `42` · splits created 2026-09-24

> **Emailing this?** `python scripts/build_submission_folder.py` collects every deliverable
> below into `Finished/` and writes a single 12 MB zip — see [README §10](README.md#10-packaging-the-submission).

> Raw images are **not** committed (~350 MB, CC-BY-NC). Download `milk10k.zip` from the ISIC
> Archive and extract it to `data/raw/milk10k/`, or point `$MILK10K_DATA` at it. Setup and the
> exact reproduction commands are in [README §6](README.md#6-setup-and-reproduction).

---

## Part A — the notebook

| File | What it is |
|---|---|
| [`homework_part_a.ipynb`](homework_part_a.ipynb) | The 12 exercises (A1.1 – A3.6), executed top to bottom with all outputs saved |
| [`homework_part_a.pdf`](homework_part_a.pdf) | The same notebook exported to PDF (35 pages, all plots visible) |

Regenerate and re-execute with:

```bash
python scripts/build_homework_notebook.py --run
python scripts/export_notebook_pdf.py homework_part_a.ipynb
```

### Where each exercise is answered

| Exercise | Section in the notebook | Supporting module |
|---|---|---|
| A1.1 Cross-check the two label files | `## A1.1` | [`src/milk10k/integrity.py`](src/milk10k/integrity.py) |
| A1.2 Five questions | `## A1.2` | — (pandas in the notebook) |
| A1.3 Lesion-level table | `## A1.3` | [`build_lesion_table`](src/milk10k/data.py) |
| A1.4 Task framing | `## A1.4` | — (written answer) |
| A2.1 Arrays, views, memory | `## A2.1` | [`src/milk10k/imaging.py`](src/milk10k/imaging.py) |
| A2.2 JPEG + file-size shortcut | `## A2.2` | [`src/milk10k/imaging.py`](src/milk10k/imaging.py) |
| A3.1 Measure the leak | `## A3.1` | [`src/milk10k/leakage.py`](src/milk10k/leakage.py) |
| A3.2 Three-way split + property test | `## A3.2` | [`src/milk10k/splits.py`](src/milk10k/splits.py) |
| A3.3 Why StratifiedGroupKFold | `## A3.3` | [`compare_fold_strategies`](src/milk10k/splits.py) |
| A3.4 Metrics and cost of errors | `## A3.4` | [`src/milk10k/metrics.py`](src/milk10k/metrics.py) |
| A3.5 Augmentation audit | `## A3.5` | [`src/milk10k/augment.py`](src/milk10k/augment.py) |
| A3.6 Lesion-level Dataset | `## A3.6` | [`src/milk10k/datasets.py`](src/milk10k/datasets.py) |

---

## Part B — the project pipeline

Everything below is rebuilt by one command:

```bash
python scripts/build_pipeline.py     # ~35 s, deterministic
```

| B | Requirement | File |
|---|---|---|
| B0 | README with task, setup, structure, data rules, decisions | [`README.md`](README.md) |
| B0 | Dependencies | [`requirements.txt`](requirements.txt) |
| B1 | Full-dataset integrity check | [`src/milk10k/integrity.py`](src/milk10k/integrity.py) → [`artifacts/image_size_summary.csv`](artifacts/image_size_summary.csv) |
| B2 | Class distribution + 11-class log scale + mapping | [`reports/figures/b2_class_distribution.png`](reports/figures/b2_class_distribution.png) |
| B2 | 3×4 gallery, one example per class | [`reports/figures/b2_class_gallery.png`](reports/figures/b2_class_gallery.png) |
| B2 | "Session 2 finding → still true? → consequence" table (6 rows, all recomputed) | [`artifacts/session2_findings_recheck.md`](artifacts/session2_findings_recheck.md) |
| B2 | Why MILK10k does not reflect real-world prevalence (310 words) | [`reports/prevalence_analysis.md`](reports/prevalence_analysis.md) |
| B3 | Label strategy + label map | [`src/milk10k/labels.py`](src/milk10k/labels.py) → [`artifacts/label_map.json`](artifacts/label_map.json) |
| B4 | Data-quality report | [`artifacts/data_quality_report.md`](artifacts/data_quality_report.md) |
| B5 | Splits, grouped by lesion, stratified | [`src/milk10k/splits.py`](src/milk10k/splits.py) |
| B5 | Train split (7,484 images) | [`artifacts/splits/train.csv`](artifacts/splits/train.csv) |
| B5 | Validation split (1,498 images) | [`artifacts/splits/val.csv`](artifacts/splits/val.csv) |
| B5 | Test split (1,498 images) | [`artifacts/splits/test.csv`](artifacts/splits/test.csv) |
| B5 | Split verification output | [`artifacts/split_verification.csv`](artifacts/split_verification.csv) |
| B6 | Preprocessing decisions (resolution, view handling) | [report §5](reports/milestone1_report.md) |
| B7 | `train_transform` / `eval_transform` in a module | [`src/milk10k/augment.py`](src/milk10k/augment.py) |
| B7 | 1 original + 7 augmentations × 3 classes | [`reports/figures/b7_augmentations.png`](reports/figures/b7_augmentations.png) |
| B7 | Augmentation table with medical justification | [`artifacts/augmentation_table.csv`](artifacts/augmentation_table.csv) |
| B8 | Dataset / DataLoaders, sampler, loud failure | [`src/milk10k/datasets.py`](src/milk10k/datasets.py) |
| B8 | Class weights (train only) | [`artifacts/class_weights.json`](artifacts/class_weights.json) |
| B8 | Normalisation statistics (train only) | [`artifacts/norm_stats.json`](artifacts/norm_stats.json) |
| B8 | Visual check: a batch of 16 after transforms | [`reports/figures/b8_train_batch.png`](reports/figures/b8_train_batch.png) |
| B9 | Milestone 1 report | [`reports/milestone1_report.md`](reports/milestone1_report.md) |
| — | Every headline number, machine-readable | [`artifacts/pipeline_summary.json`](artifacts/pipeline_summary.json) |

---

## Source code

| Module | Responsibility |
|---|---|
| [`config.py`](src/milk10k/config.py) | The single place for paths, seed and image size (`$MILK10K_DATA`) |
| [`data.py`](src/milk10k/data.py) | Loaders, class map, image table, lesion table |
| [`integrity.py`](src/milk10k/integrity.py) | Label cross-checks (A1.1), full image verification (B1) |
| [`imaging.py`](src/milk10k/imaging.py) | NumPy geometry, memory budget, PSNR, JPEG quality, file sizes (A2) |
| [`splits.py`](src/milk10k/splits.py) | Grouped/stratified lesion-level splitting (A3.2, A3.3, B5) |
| [`leakage.py`](src/milk10k/leakage.py) | The leak experiment and sibling oracle (A3.1) |
| [`metrics.py`](src/milk10k/metrics.py) | Dummy baselines, cost matrix, expected cost (A3.4) |
| [`labels.py`](src/milk10k/labels.py) | Label strategy, label map, class weights (B3) |
| [`augment.py`](src/milk10k/augment.py) | Transforms (B7) and the quantitative augmentation audit (A3.5) |
| [`datasets.py`](src/milk10k/datasets.py) | `LesionDataset` (A3.6), image `Dataset` + loaders (B8) |
| [`stats.py`](src/milk10k/stats.py) · [`color.py`](src/milk10k/color.py) · [`preprocess.py`](src/milk10k/preprocess.py) · [`loader.py`](src/milk10k/loader.py) · [`viz.py`](src/milk10k/viz.py) | Session 2 code, reused and refactored |

### Tests

```bash
python -m pytest tests/ -q      # 100 passed
```

| File | Covers |
|---|---|
| [`tests/test_splits.py`](tests/test_splits.py) | The A3.2 property test over 10 seeds, label map, class weights |
| [`tests/test_imaging_datasets.py`](tests/test_imaging_datasets.py) | PSNR, views vs copies, transforms, both Datasets, aggregation |
| [`tests/test_preprocess.py`](tests/test_preprocess.py) · [`tests/test_loader.py`](tests/test_loader.py) · [`tests/test_stats_color.py`](tests/test_stats_color.py) | Session 2 modules |

---

## Earlier sessions (context)

| Deliverable | File |
|---|---|
| Session 1 — EDA notebook | [`notebooks/01_eda_milk10k.ipynb`](notebooks/01_eda_milk10k.ipynb) · [`exercises/exercise_1.pdf`](exercises/exercise_1.pdf) |
| Session 2 — image basics | [`notebooks/02_image_basics.ipynb`](notebooks/02_image_basics.ipynb) · [`exercises/exercise_2.pdf`](exercises/exercise_2.pdf) |
| Session 2 homework — extended EDA & pipeline | [`notebooks/03_eda_pipeline.ipynb`](notebooks/03_eda_pipeline.ipynb) · [`exercises/exercise_3.pdf`](exercises/exercise_3.pdf) |
