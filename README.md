# Computer Vision & Speech Recognition — MILK10k Skin Lesion Project

Course work and project repository for **Computer Vision & Speech Recognition**.
The project builds toward an automated skin-lesion diagnosis model on the
**MILK10k** dataset. This first milestone is the exploratory data analysis (EDA).

---

## 1. The clinical task

Skin cancer is one of the most common cancers worldwide, and the outcome depends
heavily on catching malignant lesions (melanoma, basal cell carcinoma, squamous
cell carcinoma) early, while not over-treating the far more common benign
lesions (nevi, seborrheic keratoses, dermatofibromas, vascular lesions, …).

In practice a dermatologist looks at a lesion two ways:

| View | What it is | What it shows |
|---|---|---|
| **Clinical close-up** | A normal photo of the lesion and surrounding skin | Size, shape, color, context on the body, ulceration |
| **Dermatoscopic** | A magnified, polarized/immersion image through a dermatoscope | Sub-surface pigment networks, vessels, structures invisible to the naked eye |

The goal of this project is to **classify a skin lesion into its diagnostic
category from these two images plus the patient metadata** (age, sex, anatomic
site, skin tone). A model that does this well could act as a triage/decision-
support tool: flagging lesions that need a biopsy and reassuring on ones that
don't. Because the ground truth for the vast majority of MILK10k lesions is
**histopathology** (the biopsy result), the labels are as close to clinical
truth as an imaging dataset gets.

Why this is hard, and what the EDA needs to surface:

- **Severe class imbalance** — basal cell carcinoma alone is 48 % of lesions and
  five classes have fewer than 55 examples. Accuracy alone is meaningless; we
  care about per-class sensitivity, especially on the rare malignant classes.
- **Two modalities per lesion** — the clinical and dermatoscopic images carry
  complementary information and need to be handled as a pair, not as two
  independent samples.
- **Demographic and acquisition variance** — five collection sites across four
  countries, six skin-tone levels, different cameras. The model must not learn
  a site or a skin tone instead of a diagnosis.
- **Hierarchical labels** — 48 fine-grained ISIC-Dx diagnoses roll up into
  11 top-level classes; the target granularity is a modelling decision the EDA
  should inform.

## 2. The dataset — MILK10k

**MILK10k** (*Multimodal Imaging-Learning Kit*) is an open-access dataset
published on the ISIC Archive by the ViDIR group (Medical University of Vienna)
and collaborators.

| Item | Value |
|---|---|
| Lesions | 5,240 |
| Images | 10,480 (one clinical close-up + one dermatoscopic image per lesion) |
| Top-level diagnosis classes | 11 |
| Fine-grained diagnoses (ISIC-Dx) | 48 |
| Ground truth | ~95.7 % histopathology (biopsy/excision); rest by expert consensus/follow-up |
| Metadata | age (5-year bins), sex, anatomic site, skin tone (0 = very dark … 5 = very light), diagnosis, ground-truth method, ISIC ID link |
| Feature annotations | MONET-style flags: pigmentation, ulceration, erythema, vessels, hair, artifacts |
| Sources | 5 centres — Austria, Turkey, USA, North Macedonia, Australia |
| License | CC-BY-NC 4.0 (non-commercial, attribution required) |
| DOI | [10.34970/648456](https://doi.org/10.34970/648456) |

**Top-level classes (11):**

| Code | Diagnosis | Malignant? |
|---|---|---|
| `mel` | Melanoma | Yes |
| `bcc` | Basal cell carcinoma | Yes |
| `sccka` | Squamous cell carcinoma / keratoacanthoma | Yes |
| `akiec` | Actinic keratosis / intraepidermal carcinoma | Pre-malignant / in situ |
| `mal_oth` | Other malignant proliferations (incl. collision tumours) | Yes |
| `nv` | Melanocytic nevus | No |
| `bkl` | Benign keratinocytic lesion | No |
| `df` | Dermatofibroma | No |
| `vasc` | Vascular lesion / hemorrhage | No |
| `inf` | Inflammatory & infectious conditions | No |
| `ben_oth` | Other benign proliferations (incl. collision tumours) | No |

A held-out **MILK10k Benchmark** split (479 lesions / 958 images) is also
published for evaluation.

> The data is **not** committed to this repo (size + license). Download
> `milk10k.zip` from the ISIC Archive links below and extract it to
> `data/raw/milk10k/` — that folder is git-ignored.

**Files inside the download:**

| Path | Rows | What it is |
|---|---|---|
| `images/ISIC_*.jpg` | 10,480 | All images, one file per `isic_id` (clinical and dermatoscopic mixed) |
| `metadata.csv` | 10,480 | Per-image: `isic_id`, `lesion_id`, `image_type`, `age_approx`, `sex`, `anatom_site_general`, 4-level `diagnosis_1..4` hierarchy, `diagnosis_confirm_type`, `melanocytic`, … |
| `supplements/training_input.csv` | 10,480 | Per-image model inputs: `skin_tone_class`, `site`, plus 7 `MONET_*` feature scores (ulceration, hair, vessels, erythema, pigment, gel, pen ink) |
| `supplements/training_gt.csv` | 5,240 | Per-lesion one-hot ground truth over the 11 top-level classes |
| `supplements/training_supp.csv` | 10,480 | `diagnosis_full`, `diagnosis_confirm_type`, `invasion_thickness_interval` |
| `licenses/CC-BY-NC.txt`, `attribution.txt` | — | License and attribution (MILK study team) |

Images join to lesions via `lesion_id` (2 images per lesion, `image_type` ∈
{`clinical: close-up`, `dermoscopic`}).

## 3. Session 1 — Exploratory Data Analysis

Notebook: [`notebooks/01_eda_milk10k.ipynb`](notebooks/01_eda_milk10k.ipynb)
(executed, outputs included) · figures in [`reports/figures/`](reports/figures/)
· loaders in [`src/milk10k/data.py`](src/milk10k/data.py)

- [x] **Images** — count, resolution/aspect-ratio distribution, file sizes,
      colour statistics per modality, one paired example per class, mosaic
- [x] **Metadata** — completeness/missingness, age, sex, anatomic site,
      skin tone, ground-truth method, image manipulation
- [x] **Class distribution** — 11-class and 48-class counts, malignant vs
      benign ratio, imbalance ratios, diagnosis hierarchy
- [x] **Cross-tabs** — class × skin tone, class × site, class × sex,
      age by class, class × ground-truth method, MONET feature scores
- [x] **Findings** — what the EDA implies for modelling

### Key findings

| Finding | Implication |
|---|---|
| 5,240 lesions × exactly 2 images (1 clinical + 1 dermatoscopic) | Split **by lesion**, never by image; model the pair jointly |
| **72 % of lesions are malignant; BCC alone is 48 %**, NV only 14 % — a biopsy-enriched cohort, not a screening population | Prevalence-dependent metrics won't transfer; report macro-F1 / balanced accuracy / per-class sensitivity |
| Tail classes: MAL_OTH 9, BEN_OTH 44, VASC 47, INF 50, DF 52 lesions | Stratified splits, class weighting or re-sampling; consider an "other" bucket first |
| 48 fine diagnoses, most with < 10 lesions | Train at the 11-class level; use fine labels for stratification / error analysis |
| Median age 65, 60 % male, 61 % skin-tone 3, < 1 % tones 0–1 | Report metrics per tone and age band; darker skin generalisation is unproven |
| Age, site **and skin tone** vary systematically by class (keratinocyte cancers ≈ tone 3–4, older, head/neck) | Legitimate priors but potential shortcuts — audit model reliance |
| Benign classes hold the non-biopsied (clinically assessed) lesions | Benign labels are slightly noisier |
| Mixed resolutions/aspect ratios; dermatoscopy brighter and more saturated; gel/hair/ink artefacts | Aspect-preserving resize, per-modality normalisation, dermoscopy-specific augmentation |

Reproduce: `jupyter nbconvert --to notebook --execute notebooks/01_eda_milk10k.ipynb`
(≈ 10 s after the data is extracted).

## 4. Homework — Extended EDA & data pipeline

Notebook: [`notebooks/03_eda_pipeline.ipynb`](notebooks/03_eda_pipeline.ipynb)
(executed) · report: [`exercises/exercise_3.pdf`](exercises/exercise_3.pdf)
· code: the [`src/milk10k/`](src/milk10k/) package · tests: [`tests/`](tests/)

| Part | Module | What it provides |
|---|---|---|
| 1 Metadata ↔ target | `milk10k.stats` | field inventory (type / level / missingness / leakage flags), chi-square + Cramér's V, Kruskal–Wallis + ε², ranked `association_table()` |
| 2 Colour & histograms | `milk10k.color` | balanced per-class / per-modality sample, per-image RGB + grayscale histograms and statistics, class averages |
| 3 Preprocessing | `milk10k.preprocess` | `PreprocessConfig`, `preprocess_image(src) -> (array, ImageInfo)`, `preprocess_batch(items) -> BatchResult` with `skipped=[(id, reason)]` |
| 4 Data loader | `milk10k.loader` | `MILK10kLoader(df, config, batch_size, shuffle, seed)` → lazy batches of `(images, labels, ids)` from files that exist on disk |
| 5 Visualiser | `milk10k.viz` | `show_image_grid`, `plot_class_balance`, `plot_batch_summary` + the Part 1–3 plots |

```python
import milk10k
from milk10k import PreprocessConfig, MILK10kLoader, viz

df = milk10k.load_images_table()
loader = MILK10kLoader(df, PreprocessConfig(size=(128, 128)), batch_size=32, seed=0)
batch = next(iter(loader))          # batch.images (32,128,128,3) float32 [0,1], batch.labels (32,) int64
viz.show_image_grid(batch)          # sanity-check what the loader produces
viz.plot_batch_summary(batch)       # pixel-value distribution + raw vs processed
```

### Key findings

| Finding | Implication |
|---|---|
| **Age** is the most associated clinical field (ε² = 0.20; NV median 40 vs 65–70 for BCC / AKIEC / SCCKA); skin tone and anatomical site follow (V ≈ 0.15–0.18); sex is weak (V = 0.11) | Age is a legitimate auxiliary input; tone / site must be audited as centre proxies |
| `diagnosis_confirm_type` ≡ `concomitant_biopsy` (V = 0.31): every non-biopsied lesion is benign | Workflow leakage — never a model feature |
| `diagnosis_1..4`, `diagnosis_full`, `melanocytic`, `invasion_thickness_interval` are the label in disguise | Excluded from features |
| No collection-centre field exists; `site` is anatomical, and `anatom_site_general`'s 37 % "missing" = trunk | Don't impute "unknown" — it silently encodes trunk |
| Modality explains 32 % of blue-channel variance; within dermoscopy class explains only 2–8 % of channel means but **34 % of contrast** (MEL / NV / VASC dark blob vs pale BCC / AKIEC) | Colour alone separates pigmented vs keratinocytic only — MEL vs NV needs structure → learned features, per-modality normalisation |
| Class balance 280 : 1 (BCC 5,044 images vs MAL_OTH 18), 72 % malignant | Split by lesion, class weights / re-sampling, macro-F1 and per-class sensitivity |

Reproduce:

```bash
python -m pytest tests -q                                   # 32 tests
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda_pipeline.ipynb   # ≈ 20 s
python docs/build_report.py                                 # -> exercises/exercise_3.pdf
```

## 5. Homework Sessions 1-3 + Milestone 1 (current)

The graded deliverable for Sessions 1-3. **Part A** is a notebook of twelve
exercises; **Part B** turns the Session 2 code into a leak-free, reproducible
project pipeline.

| Part | Deliverable |
|---|---|
| A | [`homework_part_a.ipynb`](homework_part_a.ipynb) - 12 exercises, runs top to bottom |
| A | [`homework_part_a.pdf`](homework_part_a.pdf) - the exported notebook (35 pages) |
| B | [`scripts/build_pipeline.py`](scripts/build_pipeline.py) - rebuilds every artefact below |
| B | [`artifacts/`](artifacts/) - splits, label map, class weights, normalisation stats, reports |
| B | [`reports/milestone1_report.md`](reports/milestone1_report.md) - the 2-page Milestone 1 report |
| - | [`SUBMISSION.md`](SUBMISSION.md) - direct link to every file |

### Headline findings

- **AKIEC is the only class spanning two `diagnosis_1` values** (180 Malignant /
  123 Indeterminate), so `diagnosis_1` *cannot* be derived from an 11-class
  prediction - the two label schemes are carried side by side.
- **Biopsy enrichment is measurable in the metadata**: histopathology-confirmed
  lesions are 72.3 % malignant against 2.2 % for clinical assessment. The
  dataset's 69 % malignancy rate is a selection artefact, not a prevalence.
- **The lesion leak is worth up to 0.52 balanced-accuracy points.** A metadata-only
  random forest scores the same under a naive and a grouped split (0.424 vs
  0.425), but a "sibling oracle" that just copies the other view's label scores
  **0.857** against 0.333. The leak is harmless only to models too weak to use it.
- **Every image is JPEG quality 75**, recovered from the quantization tables.
  File size carries **no** malignancy shortcut (ROC-AUC 0.46 / 0.51).
- **The colour-jitter limits are measured, not guessed.** The real Benign/Malignant
  gap is 6.54 deg of hue and 0.077 of brightness; `hue >= 0.05` and
  `brightness >= 0.3` exceed it and are not label-safe.

---

## 6. Setup and reproduction

Python **3.13** (any 3.11+ works). Editor: VS Code with the Python and Jupyter extensions.

```bash
git clone https://github.com/jcole-jpg/Computer-Vision-Speech-Recognition.git
cd Computer-Vision-Speech-Recognition

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Where to put the data, and how the code finds it

Download `milk10k.zip` from the ISIC Archive (links in section 9) and extract it so
that `metadata.csv`, `images/` and `supplements/` sit directly inside the dataset folder:

```bash
unzip milk10k.zip -d data/raw/milk10k
```

`data/raw/milk10k/` is the **default** and needs no configuration. To keep the ~350 MB
of images elsewhere, set one environment variable - nothing else changes:

```bash
export MILK10K_DATA=/Volumes/ssd/milk10k
```

Every path in the project derives from [`src/milk10k/config.py`](src/milk10k/config.py).
There is no absolute path anywhere in the code, and no other module reads `os.environ`.
Check what the code resolved with:

```bash
python -c "from milk10k.config import describe; print(describe())"
```

### Exact commands

```bash
# 1. rebuild every Part B artefact: splits, label map, weights, stats, figures, reports
python scripts/build_pipeline.py

# 2. regenerate and execute the Part A notebook, then export it to PDF
python scripts/build_homework_notebook.py --run
python scripts/export_notebook_pdf.py homework_part_a.ipynb

# 3. run the test suite (100 tests; the real-data ones skip if MILK10k is absent)
python -m pytest tests/ -q
```

`build_pipeline.py` takes about 35 s and is deterministic: same seed in, byte-identical
splits out.

---

## 7. Repository structure

```
Computer-Vision-Speech-Recognition/
├── README.md                     ← this file
├── SUBMISSION.md                 ← direct link to every graded deliverable
├── requirements.txt              ← pinned minimum versions of every dependency
├── .gitignore                    ← keeps raw images out of git, lets artifacts/ in
├── homework_part_a.ipynb         ← PART A: the 12 exercises, executed with outputs
├── homework_part_a.pdf           ← PART A: the same notebook exported to PDF
│
├── src/milk10k/                  ← the importable package - ALL reusable logic
│   ├── config.py                 ← the ONE place for paths, seed, image size
│   ├── data.py                   ← loaders, class map, image + lesion tables
│   ├── integrity.py              ← label cross-checks (A1.1) and image verification (B1)
│   ├── imaging.py                ← NumPy geometry, PSNR, JPEG quality, file sizes (A2)
│   ├── splits.py                 ← grouped/stratified lesion-level splitting (A3.2-3, B5)
│   ├── leakage.py                ← what a lesion leak does to a real metric (A3.1)
│   ├── metrics.py                ← dummy baselines and cost-aware evaluation (A3.4)
│   ├── labels.py                 ← label strategy, label map, class weights (B3)
│   ├── augment.py                ← train/eval transforms + augmentation audit (A3.5, B7)
│   ├── datasets.py               ← LesionDataset, image Dataset, DataLoaders (A3.6, B8)
│   ├── stats.py                  ← metadata ↔ target association tests (Session 2)
│   ├── color.py                  ← histogram / colour statistics (Session 2)
│   ├── preprocess.py             ← the original preprocessing pipeline (Session 2)
│   ├── loader.py                 ← the original Session 2 loader, kept for reference
│   └── viz.py                    ← image grids, class balance, EDA plots
│
├── scripts/                      ← entry points; they orchestrate, they do not implement
│   ├── build_pipeline.py         ← builds every Part B artefact (B1-B8)
│   ├── build_homework_notebook.py← generates and executes homework_part_a.ipynb
│   └── export_notebook_pdf.py    ← notebook → HTML → headless Chrome → PDF
│
├── artifacts/                    ← COMMITTED generated outputs (small, and the evidence)
│   ├── splits/train.csv          ← 7,484 images, one row per image        (B5)
│   ├── splits/val.csv            ← 1,498 images
│   ├── splits/test.csv           ← 1,498 images
│   ├── label_map.json            ← label strategy + string→integer mapping (B3)
│   ├── class_weights.json        ← inverse-frequency weights, TRAIN only   (B8)
│   ├── norm_stats.json           ← channel mean/std, TRAIN only            (B8)
│   ├── image_size_summary.csv    ← geometry of all 10,480 images           (B1)
│   ├── data_quality_report.md    ← missing values, shortcuts, leaky columns(B4)
│   ├── augmentation_table.csv    ← each augmentation + medical justification(B7)
│   ├── split_verification.csv    ← the B5 checks, saved
│   └── pipeline_summary.json     ← every headline number, machine-readable
│
├── data/
│   ├── raw/milk10k/              ← the dataset (GIT-IGNORED - download it)
│   └── processed/                ← bulky intermediates (git-ignored)
├── notebooks/                    ← exploratory notebooks from earlier sessions
│   ├── 01_eda_milk10k.ipynb      ← Session 1 EDA
│   ├── 02_image_basics.ipynb     ← Session 2: pixels, channels, colour
│   └── 03_eda_pipeline.ipynb     ← Session 2 homework: extended EDA
├── tests/                        ← 100 pytest tests
│   ├── conftest.py               ← synthetic image fixtures
│   ├── test_splits.py            ← the A3.2 property test, 10 seeds
│   ├── test_imaging_datasets.py  ← PSNR, views, transforms, Datasets
│   └── test_*.py                 ← Session 2 loader/preprocess/stats tests
├── reports/
│   ├── milestone1_report.md      ← the Milestone 1 report (B9)
│   └── figures/                  ← every exported figure (PNG)
├── exercises/                    ← earlier graded deliverables (PDF)
└── docs/                         ← PDF builders for the earlier exercises
```

**Why it is organised this way.** The guiding rule is that **nothing importable
lives in a notebook, and nothing that runs lives only in a notebook.** Three layers:

1. `src/milk10k/` holds every reusable function, one module per concern, each named
   after the question it answers rather than the exercise that prompted it. This is
   what makes "do not copy-paste between notebooks" enforceable - the notebook
   imports `splits.split_lesions`, and so does the pipeline script, and so do the
   tests, so there is exactly one implementation to be right or wrong.
2. `scripts/` holds the entry points. They orchestrate and print; they contain no
   logic worth testing. Anything in a script that became worth reusing would move
   down into the package.
3. Generated output is kept strictly apart from source, and split by size:
   `artifacts/` is small and **committed**, because the splits and the label map
   *are* the reproducibility evidence and a grader must be able to read them
   without downloading 350 MB of JPEGs; `reports/figures/` holds PNGs;
   `data/processed/` holds bulky intermediates and is ignored.

Configuration is the fourth rule: `config.py` is the only module that knows a path
or a seed, so pointing the project at a new dataset location is a one-line change.

---

## 8. Data handling rules

| Rule | Detail |
|---|---|
| **Raw images are never committed** | `data/raw/*` is git-ignored. ~350 MB, and CC-BY-NC forbids redistribution. |
| **Split CSVs are always committed** | `artifacts/splits/*.csv` - a few hundred KB, and without them the experiment is not reproducible. |
| **Generated files live apart from source** | `artifacts/` (committed) · `reports/figures/` (committed) · `data/processed/` (ignored). |
| **Seed** | `42`, defined once in `config.SEED`. |
| **Splits created** | **2026-09-24**, by `scripts/build_pipeline.py`, from `config.SPLIT_DATE`. |
| **Normalisation & weights** | Computed on the **train split only** - never the full dataset. |
| **Regenerating** | Deterministic: same seed → byte-identical splits. |

---

## 9. Key decisions so far

Full reasoning and evidence in [`reports/milestone1_report.md`](reports/milestone1_report.md).

| Decision | Choice | Why, in one line |
|---|---|---|
| **Label strategy** | Keep `Indeterminate` as a 3rd class; keep all 11 fine classes | `P(Mal) + P(Indet)` recovers the binary decision after inference; merging at training time is irreversible. |
| **Split design** | `StratifiedGroupKFold` at lesion level, stratified on the 11-class label, 5/7-1/7-1/7 | Grouping is the only defence against the 0.857-scoring sibling oracle; the 11-class label is finer, so balancing it also balances `diagnosis_1`. |
| **Split sizes** | 1/7 (14.29 %), not 15 % | MAL_OTH's 9 lesions cap the fold count at 9, which quantises achievable sizes to 1/n; 1/7 is the nearest, and requesting it makes the realised sizes exact. |
| **Resolution** | 224 x 224 | Every image is exactly 600 x 450, so 224 is a pure downscale for 100 % of the set - no upsampling anywhere. |
| **View handling** | Train per image, **evaluate per lesion** | Doubles the training signal while keeping the metric on the unit a clinician acts on. |
| **Normalisation** | Train-split mean/std, `[0.678, 0.523, 0.471] / [0.127, 0.135, 0.157]` | Dataset statistics computed over test rows are leakage, exactly like a scaler fitted before splitting. |
| **Augmentation** | Geometry freely; `hue <= 0.02`, `brightness <= 0.1` | Measured: anything above those limits shifts colour further than the real Benign/Malignant gap. |
| **Imbalance** | Both implemented: weighted loss **and** `WeightedRandomSampler` | Train imbalance is 300:1 (BCC 3,604 vs MAL_OTH 12). Use one or the other, not both. |

---

## 10. Packaging the submission

Everything a grader needs, collected into one folder and one emailable zip:

```bash
python scripts/build_submission_folder.py
```

Writes `Finished/` (~15 MB, 59 files) and `Cole_Joshua_CVSR_Sessions_1-3.zip` (~12 MB):

```
Finished/
├── 00_START_HERE.pdf / .md      cover sheet: repo URL + where every requirement is answered
├── SUBMISSION.md                the required link list
├── 01_Part_A_notebook/          homework_part_a.pdf (33 pp) + .ipynb
├── 02_Part_B_milestone1/        report, prevalence analysis, README, artifacts, figures, source
└── 03_earlier_sessions/         exercises 1-3
```

Both are git-ignored: every file in them is assembled from content already committed
elsewhere in the repo, so committing the bundle would duplicate ~15 MB. The Markdown
reports are rendered to PDF (headless Chrome, no LaTeX needed) and the photo-heavy
figures are re-encoded to JPEG so the zip stays comfortably under a 25 MB mail limit —
full-resolution PNGs remain in `reports/figures/`.

---

## 11. References


- ISIC Archive — MILK10k dataset page: https://api.isic-archive.com/doi/milk10k/
- ISIC Archive — MILK10k collection (10,480 images): https://api.isic-archive.com/collections/425/
- ISIC Archive — MILK10k Benchmark: https://api.isic-archive.com/doi/milk10k-benchmark/
- ISIC Challenge — MILK10k Benchmark landing page: https://challenge.isic-archive.com/landing/milk10k/
- Tschandl P. et al., *MILK10k: A Hierarchical Multimodal Imaging-Learning
  Toolkit for Diagnosing Pigmented and Nonpigmented Skin Cancer and its
  Simulators*, Journal of Investigative Dermatology (2025):
  https://www.sciencedirect.com/science/article/pii/S0022202X25022705
- *Comparative and complementary diagnostic value of dermatoscopy and clinical
  close-up photography in skin cancer diagnosis: A study from the MILK10k
  dataset*, JAAD (2026): https://www.jaad.org/article/S0190-9622(26)00479-2/fulltext

---

**Author:** Joshua Cole · GitHub [@jcole-jpg](https://github.com/jcole-jpg)
