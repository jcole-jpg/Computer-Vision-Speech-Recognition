# Milestone 1 — Data Pipeline

**MILK10k skin-lesion classification** · Joshua Cole · seed 42 · splits created 2026-09-24

Every number is produced by `scripts/build_pipeline.py` and re-read from `artifacts/`.
Companions: `prevalence_analysis.md` (B2), `artifacts/data_quality_report.md` (B4).

## 1. Label strategy

Primary target `diagnosis_1`, **all three classes kept**: Benign 1,483 (28.3%), Malignant
3,634 (69.4%), Indeterminate 123 (2.3%).

Keeping Indeterminate is not obvious: all 123 are AKIEC, so the label records *pathologist
confidence*, not a distinct lesion — and AKIEC is also 180 Malignant lesions. The decision
rests on **reversibility**: a 3-class model collapses to the binary decision at any time via
`P(Malignant) + P(Indeterminate)`, whereas a model trained on merged labels can never be
un-merged. Keeping the finer label costs nothing and preserves the option. AKIEC is also the
only class spanning two `diagnosis_1` values, so `diagnosis_1` **cannot** be derived from an
11-class prediction; both schemes are carried side by side.

The same argument keeps **all 11 fine classes**, MAL_OTH's 9 lesions included: merging rare
classes into "other" is irreversible and yields a category no clinician can act on, and
imbalance belongs in the weights, not the label definition. Caveat recorded in
`label_map.json`: MAL_OTH cannot be *evaluated* at 9 lesions (~1 reaches test, so recall is
0.0 or 1.0), so its metric is reported with its support and excluded from the macro-average.

## 2. Split design and verification

Splitting is at **lesion level** with `StratifiedGroupKFold`, stratified on the 11-class label
(the finer label, so `diagnosis_1` balances for free); images are assigned afterwards.

**Why grouping is non-negotiable.** A metadata-only random forest scores the same under a
naive and a grouped split (0.424 ± 0.007 vs 0.425 ± 0.003 balanced accuracy) — the leak
*looks* harmless. It is not: a "sibling oracle" that merely copies the other view's label
scores **0.857** under the naive split (81.4% of test images have their sibling in train)
against 0.333 under the grouped one. The forest's four coarse features cannot identify an
individual lesion; a CNN seeing two photos of the same mole can. Measured absence of inflation
is a property of the weak model, not of the data.

**Why 1/7, not 15%.** A fold splitter holds out `1/n`, and asking for more folds than the
rarest class has members guarantees a fold with none of it. MAL_OTH's 9 lesions cap `n_splits`
at 9, quantising achievable sizes to 1/n. 1/7 is nearest 15%; requesting 0.15 leaves train
1.4 pp off, requesting 1/7 makes the sizes exact.

| Verification (`artifacts/split_verification.csv`) | Result |
|---|---|
| Lesion overlap between every pair of splits | **0 / 0 / 0** |
| Every lesion has exactly 2 images, all in one split | **PASS** (5,240 / 5,240) |
| Sizes: 3,742 / 749 / 749 lesions (7,484 / 1,498 / 1,498 images) | 71.41 / 14.29 / 14.29 % |
| Max deviation from requested size | **0.02 pp** (tolerance 1 pp) |
| Max class-proportion deviation — 11-class / `diagnosis_1` | **0.10 pp / 0.14 pp** |
| Property test: 10 seeds × (disjoint, sizes, reproducible) | **PASS** |
| `StratifiedKFold` (images) / `GroupKFold` / **`StratifiedGroupKFold`** | 8,316 leaks / 3.63 pp drift / **0 and 0.10 pp** |

## 3. Imbalance on TRAIN

**300.3 : 1** on the 11-class scheme (BCC 3,604 images vs MAL_OTH 12); 29.2 : 1 on
`diagnosis_1` (Malignant 5,190 vs Indeterminate 178). Both remedies are implemented from the
**train split only** (`class_weights.json`): inverse-frequency loss weights `N / (K·n_c)`
(MAL_OTH 56.70, BCC 0.19) and a `WeightedRandomSampler`. They are alternatives, not a stack —
each corrects the imbalance once. The sampler demonstrably works: over 20 training batches the
label histogram is **213 / 221 / 206**, essentially balanced despite the 29:1 raw ratio.

## 4. Quality issues found and how they were handled

- **Integrity is clean.** All 10,480 rows resolve to a file, `Image.verify()` passes on every
  one, all images are exactly 600 × 450, and no two share an MD5.
- **`anatom_site_general`'s 37.3% "missing" is really `trunk`.** 3,850 of the 3,912 missing
  rows have `site == "trunk"`; only **62 (0.6%)** are genuinely unknown — the column has no
  trunk category. Both obvious rules ("impute the mode", "encode as unknown") would collapse
  the most common anatomical site into a meaningless bucket. `data.resolve_site()` merges the
  two columns, dropping missingness from 37.3% to 0.6%. Lesson: "missing" in a medical table
  often means "recorded in another column".
- **`diagnosis_confirm_type` excluded** — histopathology lesions are 72.3% malignant vs 2.2%
  for clinical assessment; it encodes prior suspicion, downstream of the label.
- **`image_manipulation` excluded** — the "altered" rate runs 0.0% (MAL_OTH) to 18.2%
  (BEN_OTH), so it carries class signal, but it describes post-processing, not skin.
- **No file-size shortcut** — ROC-AUC of raw file size is 0.464 dermoscopic / 0.511 clinical,
  i.e. chance. A useful *negative* result: a dataset whose malignant lesions came from a
  different device would fail this, and a CNN would find that before it found biology.
- **All images are JPEG quality 75** (recovered from the quantization tables), so intermediates
  must never be re-saved as JPEG — a second quantization adds fresh error.
- **Loaders fail loudly.** A missing file raises `MissingImageError`; silently skipping rows
  shrinks the dataset and shifts the class balance without warning.

## 5. Preprocessing choices

- **224 × 224.** Every image is exactly 600 × 450, so 224 is a pure downscale for 100% of the
  set — no invented detail. Evaluation resizes the short side to 256 then centre-crops,
  preserving the 4:3 ratio rather than distorting it.
- **Normalisation from the train split only**: mean `[0.678, 0.523, 0.471]`, std
  `[0.127, 0.135, 0.157]` over 500 training images. Computing these over the full dataset leaks
  test information exactly as a scaler fitted before splitting does; they differ markedly from
  ImageNet's — skin is redder and lower-variance than natural images.
- **Train per image, evaluate per lesion.** Each view is independently informative, so training
  on images doubles the signal; but the label belongs to the lesion, so `aggregate_predictions`
  averages the two views before any metric. The alternatives — one image type only (throws away
  half the data) and a two-stream model (better, a Milestone 2 decision) — are supported as far
  as `LesionDataset`, which already returns both views. **Evaluating per image would be a
  leak**: the two views are not independent, so the effective sample is 749 lesions, not 1,498
  images; treating them as independent nearly halves every confidence interval and lets a model
  score 50% on a lesion it answered two different ways.
- **Augmentation with measured limits.** Geometry freely (flips, ±30°, `RandomResizedCrop`
  `scale=(0.8, 1.0)` — skin has no canonical orientation, asymmetry is rotation-invariant, and
  the 0.8 floor keeps the diagnostic border in frame). Colour capped at `hue = 0.02`,
  `brightness = 0.1`, and those numbers are **measured**: the real Benign↔Malignant gap is
  6.54° of hue and 0.077 of brightness, while `hue = 0.05` shifts 16.07° (2.5×), `hue = 0.5`
  shifts 178° (27×) and `brightness = 0.3` shifts 0.174 (2.3×). Anything above the caps changes
  colour more than the class difference itself. `eval_transform` is asserted bit-identical
  across two applications.

## 6. What could still go wrong

The grouped split closes the two-views-per-lesion channel and the MD5 check rules out exact
duplicates, but **nothing rules out near-duplicates** — two lesions from one patient in one
session, sharing skin texture, hair and background. MILK10k ships no `patient_id`, so
patient-level grouping is impossible with the available metadata: a known, unclosable gap at
Milestone 1, to revisit with embedding-based detection before any headline number is quoted.
The deeper issue is **biopsy enrichment** (`prevalence_analysis.md`): 69.4% of lesions are
malignant only because a lesion enters MILK10k only if someone was suspicious enough to excise
it, so every accuracy figure describes performance *on lesions already selected for biopsy*,
and precision will collapse under a real base rate. Further uncorrected biases: five centres in
four countries, a skin-tone distribution skewed toward lighter skin, one pathology culture's
AKIEC labelling. **A clinician should know** that (1) this is a triage aid for lesions already
judged worth careful photography, silent about lesions never presented; (2) its confidence is
calibrated to a 69%-malignant population and will be badly over-confident in a normal clinic
until recalibrated; (3) rare-class performance — MAL_OTH above all, 9 lesions in the entire
dataset — is not measurable and must not be reported as if it were; (4) both views come from
study-grade equipment at fixed resolution and a single JPEG setting, so smartphone performance
is untested; and (5) Indeterminate means the pathologist hedged, so a model predicting it is
predicting that hedge — a signal to escalate to a human, not a diagnosis.
