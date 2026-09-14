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

## 3. Milestone 1 — Exploratory Data Analysis

Notebook: [`notebooks/01_eda_milk10k.ipynb`](notebooks/01_eda_milk10k.ipynb)
(executed, outputs included) · figures in [`reports/figures/`](reports/figures/)
· loaders in [`src/milk10k.py`](src/milk10k.py)

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

## 4. Project structure

```
Computer-Vision-Speech-Recognition/
├── README.md                ← this file
├── requirements.txt         ← Python dependencies
├── .gitignore               ← keeps data / checkpoints / secrets out of git
├── data/
│   ├── raw/
│   │   └── milk10k/         ← extracted dataset (git-ignored)
│   │       ├── images/      ← 10,480 × ISIC_*.jpg
│   │       ├── metadata.csv
│   │       └── supplements/ ← training_input / training_gt / training_supp .csv
│   └── processed/           ← derived tables / cached stats (git-ignored)
├── notebooks/
│   ├── 01_eda_milk10k.ipynb ← Exercise 1 EDA (executed)
│   └── 02_image_basics.ipynb ← Session 2: pixels, channels, colour (executed)
├── src/
│   └── milk10k.py           ← loaders, class map, merged image/lesion tables
├── reports/
│   └── figures/             ← 12 exported EDA figures (PNG)
├── exercises/
│   ├── exercise_1.pdf       ← deliverable: repo link + editor screenshot
│   └── exercise_2.pdf       ← deliverable: session 2 notebook as PDF
└── docs/
    ├── build_deliverable.py ← builds exercises/exercise_1.pdf
    └── editor_screenshot.png
```

## 5. Setup

Editor: **VS Code** with the Python and Jupyter extensions.

```bash
git clone https://github.com/jcole-jpg/Computer-Vision-Speech-Recognition.git
cd Computer-Vision-Speech-Recognition

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download milk10k.zip from the ISIC Archive (see links below), then:
unzip milk10k.zip -d data/raw/milk10k

jupyter lab notebooks/
```

## 6. Deliverables

| Deliverable | Where |
|---|---|
| Public GitHub repository with the EDA code | https://github.com/jcole-jpg/Computer-Vision-Speech-Recognition |
| EDA notebook | [`notebooks/01_eda_milk10k.ipynb`](notebooks/01_eda_milk10k.ipynb) |
| PDF with repo link + editor screenshot of the project structure | [`exercises/exercise_1.pdf`](exercises/exercise_1.pdf) |
| Session 2 notebook — image basics (pixels, channels, colour) | [`notebooks/02_image_basics.ipynb`](notebooks/02_image_basics.ipynb) · [`exercises/exercise_2.pdf`](exercises/exercise_2.pdf) |

## 7. References

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
