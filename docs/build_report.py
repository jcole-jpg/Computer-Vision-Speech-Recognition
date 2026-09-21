"""Build the homework report PDF (Extended EDA & data pipeline) -> exercises/exercise_3.pdf

Usage:
    python docs/build_report.py [output.pdf]

Numbers come from the CSVs written by notebooks/03_eda_pipeline.ipynb
(data/processed/hw_*.csv) and the figures from reports/figures/, so run the
notebook first. Renders HTML and prints it with headless Chrome.
"""

from __future__ import annotations

import base64
import datetime as dt
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
PROC = ROOT / "data" / "processed"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO_URL = "https://github.com/jcole-jpg/Computer-Vision-Speech-Recognition"


def img(name: str, width: str = "100%", caption: str = "") -> str:
    path = FIG / name
    uri = f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure style="width:{width}"><img src="{uri}" alt="{name}">{cap}</figure>'


def association_rows() -> str:
    s = pd.read_csv(PROC / "hw_association_summary.csv")
    s = s[s.field != "image_type"]
    rows = []
    for r in s.itertuples():
        flag = ("bias" if str(r.note).startswith("BIAS") else
                "image-derived" if str(r.field).startswith("MONET") else
                "sparse" if r.pct_missing >= 30 else "")
        eff = f"V = {r.effect_size:.2f}" if r.effect_name == "Cramér's V" else f"ε² = {r.effect_size:.2f} (√ = {r.effect_r:.2f})"
        field = r.field.replace("MONET_gel_water_drop_fluid_dermoscopy_liquid", "MONET_gel").replace(
            "MONET_skin_markings_pen_ink_purple_pen", "MONET_pen_ink")
        stat = ("χ²" if r.test == "chi-square" else "H") + f" = {r.statistic:,.0f}"
        rows.append(f"<tr><td><code>{field}</code></td><td>{r.kind} · {r.level}</td>"
                    f"<td class='num'>{r.pct_missing:.0f}%</td><td class='num'>{r.n:,}</td>"
                    f"<td class='num'>{stat}</td><td class='num'>{r.p_value:.0e}</td>"
                    f"<td class='num'>{eff}</td><td class='flag'>{flag}</td></tr>")
    return "\n".join(rows)


def color_effect_rows() -> str:
    e = pd.read_csv(PROC / "hw_color_effect_sizes.csv", index_col=0)
    names = {"gray_mean": "grayscale mean", "gray_std": "grayscale std (contrast)", "r_mean": "R mean",
             "g_mean": "G mean", "b_mean": "B mean"}
    out = []
    for k in ["gray_mean", "gray_std", "r_mean", "g_mean", "b_mean"]:
        r = e.loc[k]
        out.append(f"<tr><td>{names[k]}</td><td class='num'>{r['class | dermoscopic']:.2f}</td>"
                   f"<td class='num'>{r['class | clinical']:.2f}</td><td class='num'>{r['modality']:.2f}</td></tr>")
    return "\n".join(out)


def color_summary_rows() -> str:
    d = pd.read_csv(PROC / "hw_color_summary_dermoscopic.csv", index_col=0)
    c = pd.read_csv(PROC / "hw_color_summary_clinical.csv", index_col=0)
    out = []
    for cls in d.index:
        out.append(f"<tr><td><code>{cls}</code></td><td class='num'>{int(d.loc[cls,'n_images'])}</td>"
                   f"<td class='num'>{d.loc[cls,'gray_mean_mean']:.0f}</td><td class='num'>{d.loc[cls,'gray_std_mean']:.0f}</td>"
                   f"<td class='num'>{d.loc[cls,'r_mean_mean']:.0f} / {d.loc[cls,'g_mean_mean']:.0f} / {d.loc[cls,'b_mean_mean']:.0f}</td>"
                   f"<td class='num'>{c.loc[cls,'gray_mean_mean']:.0f}</td><td class='num'>{c.loc[cls,'gray_std_mean']:.0f}</td>"
                   f"<td class='num'>{c.loc[cls,'r_mean_mean']:.0f} / {c.loc[cls,'g_mean_mean']:.0f} / {c.loc[cls,'b_mean_mean']:.0f}</td></tr>")
    return "\n".join(out)


def build_html() -> str:
    today = dt.date.today().strftime("%d %B %Y")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 11mm 12mm 11mm 12mm; }}
  body {{ font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 9.1pt;
         line-height: 1.3; color: #0b0b0b; margin: 0; }}
  h1 {{ font-size: 17pt; margin: 0 0 2px; }}
  h2 {{ page-break-after: avoid; font-size: 11.5pt; margin: 8px 0 3px; padding-bottom: 2px; border-bottom: 1.5px solid #0b0b0b; }}
  h3 {{ font-size: 10.2pt; margin: 8px 0 3px; }}
  p {{ margin: 3px 0 5px; text-align: justify; }}
  ul {{ margin: 2px 0 5px 16px; padding: 0; }} li {{ margin: 1px 0; }}
  .meta {{ color: #52514e; font-size: 9pt; margin-bottom: 6px; }}
  .meta a {{ color: #2a78d6; text-decoration: none; }}
  code {{ font-family: "SF Mono", Menlo, monospace; font-size: 8.4pt; background: #f2f1ee; padding: 0 3px; border-radius: 3px; }}
  pre {{ font-family: "SF Mono", Menlo, monospace; font-size: 7.2pt; line-height: 1.3; background: #f6f5f2; border: 1px solid #e1e0d9;
        padding: 5px 7px; border-radius: 4px; margin: 3px 0 5px; white-space: pre; overflow: hidden; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 7.3pt; margin: 2px 0 4px; }}
  th, td {{ padding: 0.5px 4px; border-bottom: 1px solid #e1e0d9; text-align: left; vertical-align: top; }}
  th {{ font-weight: 600; color: #52514e; border-bottom: 1.2px solid #c3c2b7; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  td.flag {{ color: #eb6834; font-weight: 600; }}
  figure {{ margin: 4px 0 6px; page-break-inside: avoid; }}
  figure img {{ width: 100%; display: block; }}
  figcaption {{ font-size: 7.6pt; color: #52514e; margin-top: 2px; }}
  .row {{ display: flex; gap: 10px; align-items: flex-start; page-break-inside: avoid; }}
  table {{ page-break-inside: avoid; }}
  .center {{ margin: 4px auto 6px; }}
  .row > * {{ flex: 1; min-width: 0; }}
  .box {{ border: 1px solid #e1e0d9; border-radius: 5px; padding: 6px 9px; background: #fbfaf8; }}
  .callout {{ border-left: 3px solid #2a78d6; padding: 3px 8px; margin: 5px 0; background: #f4f8fd; }}
  .pb {{ page-break-before: always; }}
</style></head><body>

<h1>Extended EDA &amp; Data Pipeline — MILK10k</h1>
<div class="meta">Computer Vision &amp; Speech Recognition · Homework (Sessions 1–2 follow-up) · Joshua Cole · {today}<br>
Code: <a href="{REPO_URL}">{REPO_URL}</a> — package <code>src/milk10k/</code>
(<code>stats</code>, <code>color</code>, <code>preprocess</code>, <code>loader</code>, <code>viz</code>), driver notebook
<code>notebooks/03_eda_pipeline.ipynb</code>, tests <code>tests/</code> (32 pytest cases).</div>

<h2>Part 1 — Metadata analysis: correlation with the target</h2>
<p><b>Setup.</b> Target = the 11-class top-level diagnosis (<code>label</code> from <code>training_gt.csv</code>). The 26 non-id,
non-target columns were typed and their missingness measured (table in the notebook). Two are constants, none is free text, and
seven are <i>the label in disguise</i> (<code>diagnosis_1..4</code>, <code>diagnosis_full</code>, <code>melanocytic</code>,
<code>invasion_thickness_interval</code>) — excluded from testing and flagged as leakage. Contrary to what the dataset paper suggests,
there is <b>no collection-centre or device field</b>: <code>site</code> is a fine-grained <i>anatomical</i> site, and
<code>anatom_site_general</code> is its coarser copy whose 37&nbsp;% "missing" rows are exactly the trunk lesions (missing-not-at-random —
imputing "unknown" would silently encode <i>trunk</i>).</p>
<p><b>Unit of analysis and statistics.</b> Every lesion has two images with identical patient fields, so lesion-level fields are tested on
the 5,240 lesions, not the 10,480 images (which would double every count). Categorical field → Pearson <b>chi-square</b> test of
independence + bias-corrected <b>Cramér's V</b>; numeric / ordinal field → <b>Kruskal–Wallis H</b> + <b>ε²</b>. Kruskal–Wallis was chosen
over one-way ANOVA because age is 5-year-binned and skewed, the MONET scores are bounded in [0, 1] and non-normal, and class sizes run
from 9 to 2,522 — the F-test's assumptions fail while the rank test stays valid. With n in the thousands every p-value is ≈ 0, so fields
are ranked on effect size; because V is correlation-scale and ε² is variance-scale, √ε² is used for the common ranking.</p>

<table>
<tr><th>field</th><th>type · level</th><th>missing</th><th>n</th><th>statistic</th><th>p</th><th>effect size</th><th>caveat</th></tr>
{association_rows()}
</table>

<div class="row">
{img("09_field_association_ranking.png", "52%", "Fig. 1 — Effect size of every usable field, common scale.")}
{img("09_num_age_approx.png", "48%", "Fig. 2 — Age by class: NV median 40 vs 65–70 for BCC / AKIEC / SCCKA.")}
</div>
<div class="row">
{img("09_cat_skin_tone_class.png", "50%", "Fig. 3 — P(class | skin tone): tone 2 is 37 % NV / 24 % MEL, tones 3–5 ≈ 50 % BCC.")}
{img("09_cat_diagnosis_confirm_type.png", "50%", "Fig. 4 — Ground-truth method: every non-biopsied lesion is benign — a workflow leak.")}
</div>

<p><b>Findings.</b> The <b>most associated usable fields</b> are (1) <b><code>age_approx</code></b> (ε² = 0.20 — median 40 for NV vs 65–70
for the keratinocyte cancers; genuine clinical prior knowledge and a legitimate auxiliary input), (2) the image-derived
<b><code>MONET_pigmented</code></b> (ε² = 0.39) and <b><code>MONET_erythema</code></b> (0.24) — the strongest signals of all, but they are
outputs of a vision-language model run on the image, i.e. image features rather than clinical metadata — and (3)
<b><code>skin_tone_class</code></b> / <b><code>site</code></b> (V ≈ 0.15–0.18: hands are 50&nbsp;% SCCKA, feet 65&nbsp;% BCC).
<b>Least useful:</b> <code>image_type</code> (V = 0 by construction), <code>MONET_hair</code> (ε² = 0.02), <code>sex</code>
(V = 0.11: 60&nbsp;% male overall, 70&nbsp;% in SCCKA but 50&nbsp;% in NV) and <code>anatom_site_special</code> (V = 0.70 looks large but
n = 103, 98&nbsp;% missing — unusable).</p>
<div class="callout"><b>Bias / leakage flags.</b> <code>diagnosis_confirm_type</code> ≡ <code>concomitant_biopsy</code> (identical
χ² = 514, V = 0.31): whether a lesion was biopsied is a decision made <i>because of</i> the suspected diagnosis, and every non-biopsied
lesion is benign — a model given it learns the clinician's workflow, not the lesion. <code>image_manipulation</code> (V = 0.13): 327 of
335 "altered" images are clinical photos — an acquisition artefact correlated with class. <code>skin_tone_class</code> (only 2&nbsp;% of
lesions are tone 0–1, 61&nbsp;% tone 3) and <code>site</code> are legitimate priors but also proxies for the contributing centre and
patient population; their use must be audited with per-tone / per-site metrics.</div>

<h2>Part 2 — Colour and histogram analysis across the dataset</h2>
<p><b>Sample.</b> 30 images per class <i>per modality</i>, seed 42 (MAL_OTH keeps all 9), 618 images in total; per-image grayscale and
R/G/B histograms (64 bins) and mean / std are computed on the raw 0–255 image through the same decode path as the preprocessing
pipeline (<code>milk10k.color</code>). The two modalities are never pooled: the dermoscopic / clinical gap dominates colour, so a fair
class comparison must be made within one modality.</p>
{img("10_hist_overlay_dermoscopic.png", "100%", "Fig. 5 — Dermoscopic images: class-average grayscale and R / G / B histograms (benign = blues, dashed; malignant = oranges, solid). MEL, NV and VASC carry a broad dark tail; BCC, AKIEC and INF are a single bright peak.")}
{img("10_color_boxplots_dermoscopic.png", "100%", "Fig. 6 — Dermoscopic images: per-image colour statistics by class. Channel means overlap; contrast (grayscale std) separates the pigmented classes.")}
<div class="row">
<div>
<table>
<tr><th rowspan="2">class</th><th rowspan="2">n</th><th colspan="3">dermoscopic</th><th colspan="3">clinical</th></tr>
<tr><th>gray</th><th>std</th><th>R / G / B</th><th>gray</th><th>std</th><th>R / G / B</th></tr>
{color_summary_rows()}
</table>
<figcaption>Table 2 — Per-class mean of the per-image statistics (0–255).</figcaption>
</div>
<div style="flex:0.62">
<table>
<tr><th>statistic</th><th>ε² class<br>(dermoscopic)</th><th>ε² class<br>(clinical)</th><th>ε²<br>modality</th></tr>
{color_effect_rows()}
</table>
<figcaption>Table 3 — Share of variance of each per-image statistic explained by class (within one modality) vs by modality (Kruskal–Wallis ε²).</figcaption>
</div>
</div>
<p><b>What differs.</b> <b>Modality dominates</b>: dermoscopic images are bluer and brighter (mean B ≈ 140 vs ≈ 105 in clinical photos;
modality alone explains 32&nbsp;% of B-channel variance) because of the polarised / immersion optics and white-balanced light, whereas
clinical photos are warm skin-tone images under room light. <b>Within dermoscopy, brightness and channel means barely separate classes</b>
(class explains 2–8&nbsp;% of their variance; every class mean sits between 145 and 161) — but <b>contrast does</b>: grayscale std has
ε² = 0.34, with MEL (37), NV (39) and VASC (36) — pigmented / vascular lesions that put a dark blob on light skin — against BCC (14.5),
AKIEC (17) and INF (16), pale lesions with one sharp peak near 160. In clinical photos the pattern flips: contrast is uninformative
(ε² = 0.03) while G / B means carry 13–15&nbsp;%: AKIEC / SCCKA photos are darker and redder (gray ≈ 130, B ≈ 93) than VASC / NV / BCC
(≈ 146–152, B ≈ 111–126).</p>
<p><b>Alternative explanations.</b> (i) <i>Confounding with skin tone and site</i> (Part 1): BCC / SCCKA / AKIEC come from older,
tone-3–5 patients on sun-damaged head / neck / hand skin, whose <i>background</i> skin is redder and more textured regardless of the
lesion, while NV / MEL cluster in tone 2 on the trunk. (ii) <i>Acquisition</i>: dermatoscope model, contact gel, polarisation and
white balance differ between the five centres, which also have different case mixes; the dark vignette in some frames lowers the mean
with no lesion involvement. (iii) <i>Framing</i>: clinical close-ups are not standardised — a 3&nbsp;mm nevus fills 2&nbsp;% of the frame, an
ulcerated SCC 40&nbsp;% — so the "image mean" mostly measures how much skin is in frame. (iv) <i>Sample</i>: the within-class spread of
the means (std 13–28) is as large as the between-class differences, and MAL_OTH has 9 images.</p>
<div class="callout"><b>Early verdict.</b> Colour / intensity alone will not classify 11 classes: class means overlap almost completely and
MEL vs NV — the clinically critical pair — are indistinguishable on every simple statistic. These features do give a coarse partition,
<b>pigmented / vascular (NV, MEL, VASC — high contrast) vs keratinocytic (BCC, AKIEC, SCCKA, BKL — low contrast)</b>. The real signal
must come from <i>structure</i> (pigment network, vessels, borders, texture) and from both modalities together: a learned feature
extractor (CNN / ViT) with the modality kept explicit, age as an auxiliary input, and per-modality normalisation so that the modality
gap is not the first thing the network learns.</div>

<h2>Parts 3–5 — Preprocessing, data loader and visualiser: design</h2>
<div class="row">
<div style="flex:1.15">
<p><b>One config, one processing path.</b> <code>PreprocessConfig(size, color_mode, resize_mode, normalize, mean, std)</code> is an
immutable, validated dataclass; <code>preprocess_image(src, config) → (array, ImageInfo)</code> is the only place pixels are decoded,
resized and normalised, and the colour analysis, batch function, loader and visualiser all call it, so any change happens once.
Resize defaults to an aspect-preserving <b>centre crop</b> (MILK10k mixes aspect ratios; stretching warps lesion geometry).
Normalisation defaults to <b>global min–max (÷255 → [0, 1])</b>, deliberately <i>not</i> per-image min–max, which would stretch every
image to full contrast and erase exactly the brightness / contrast differences Part 2 measured; <code>zscore</code> with dataset-level
mean / std is available for training. Every result carries an <code>ImageInfo</code> (input / output shape, dtype, value range, steps),
so an array is never a mystery.</p>
<p><b>Failures are data, not exceptions.</b> <code>preprocess_batch(items, config) → BatchResult(images (N,H,W,C), ids, infos,
skipped=[(id, reason)])</code> catches per-item errors and continues; in the notebook a missing path and a text file are reported as
<code>FileNotFoundError</code> / <code>UnidentifiedImageError</code> while the four good images are stacked.</p>
<p><b>Loader = table + config + batch size.</b> <code>available_subset()</code> drops rows with no file on disk at construction
(<code>loader.n_unavailable</code>), labels are encoded in <code>CLASS_CODES</code> order, and nothing is read until iteration. Because
it takes any DataFrame with <code>isic_id / path / label</code>, one class serves a dermoscopic-only loader, a train / val split or a
debugging subset by filtering the table first. <b>Visualiser accepts what the loader emits</b>: <code>show_image_grid</code> and
<code>plot_batch_summary</code> take a <code>Batch</code> directly, <code>plot_class_balance</code> a DataFrame or a loader, so the
sanity check is one line after building a loader. Colour encodes only benign vs malignant; eleven classes are identified by position.</p>
</div>
<div class="box" style="flex:0.95">
<b>Loader interface</b>
<pre>cfg = PreprocessConfig(size=(128, 128))
loader = MILK10kLoader(df, cfg, batch_size=32,
        shuffle=True, seed=0, label_col="label")
loader.n_images, loader.n_unavailable  # 10480, 0
loader.classes      # ['AKIEC','BCC',...,'VASC']
len(loader)         # 328 batches per epoch
for batch in loader:          # lazy, on the fly
    batch.images    # (32,128,128,3) float32
    batch.labels    # (32,) int64 -> classes
    batch.label_names, batch.ids  # aligned
    batch.skipped   # [(id, reason)]
loader.class_counts()  # images per class</pre>
<b>Tests</b>: <code>python -m pytest tests -q</code> — 32 cases: shapes / ranges for every config, bad-file skipping,
availability filtering, label alignment after a skip, seeded shuffling, statistics on synthetic tables, real MILK10k smoke tests.
</div>
</div>
<div class="row">
<div style="flex:1.05">
{img("11_before_after.png", "100%", "Fig. 7 — Raw vs processed for four real images under four configs: 224 RGB centre-crop min–max, 224 RGB stretch, 128 grayscale, 224 RGB z-score (re-stretched for display).")}
</div>
<div style="flex:0.95">
{img("12_class_balance_images.png", "100%", "Fig. 8 — Class balance of the available images: BCC 48 %, MAL_OTH 18 images; 280 : 1.")}
{img("12_batch_summary_zscore.png", "100%", "Fig. 9 — Batch summary of a z-score loader batch: mean 0.000, std 1.000, plus raw vs processed.")}
</div>
</div>
<p><b>Class balance.</b> All 10,480 images are available locally. The data is <b>severely imbalanced</b>: BCC is 48&nbsp;% of images, five
classes have ≤ 104 images (MAL_OTH 18), a 280 : 1 largest-to-smallest ratio, and 72&nbsp;% of lesions are malignant. For training this
means: split <b>by lesion</b> and stratify by class; use class-weighted loss or re-sampling (the loader's <code>class_counts()</code>
gives the weights) or merge the five tail classes into an "other" bucket first; and evaluate with macro-F1 / balanced accuracy and
per-class sensitivity — a model that always says BCC already scores 48&nbsp;% accuracy.</p>
</body></html>"""


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "exercises" / "exercise_3.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(build_html(), encoding="utf-8")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out}", page.as_uri()], check=True, capture_output=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
