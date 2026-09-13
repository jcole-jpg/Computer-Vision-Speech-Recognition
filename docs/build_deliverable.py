"""Build the Milestone 1 deliverable PDF (repo link + editor screenshot).

Usage:
    python docs/build_deliverable.py [screenshot.png] [output.pdf]

Defaults: docs/editor_screenshot.png -> docs/milestone1_deliverable.pdf
Renders an HTML page and prints it to PDF with headless Chrome.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO_URL = "https://github.com/jcole-jpg/Computer-Vision-Speech-Recognition"

STRUCTURE = """\
Computer-Vision-Speech-Recognition/
├── README.md                     project overview, clinical task, dataset, findings
├── requirements.txt              Python dependencies
├── .gitignore                    keeps data / venv / checkpoints out of git
├── data/
│   ├── raw/milk10k/              extracted dataset (git-ignored)
│   │   ├── images/               10,480 x ISIC_*.jpg
│   │   ├── metadata.csv
│   │   └── supplements/          training_input / training_gt / training_supp .csv
│   └── processed/                derived tables (git-ignored)
├── notebooks/
│   └── 01_eda_milk10k.ipynb      Milestone 1 EDA (executed, outputs included)
├── src/
│   └── milk10k.py                loaders, class map, merged image / lesion tables
├── reports/
│   └── figures/                  12 exported EDA figures
└── docs/
    ├── build_deliverable.py      this script
    ├── editor_screenshot.png
    └── milestone1_deliverable.pdf"""


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_html(screenshot: Path) -> str:
    fig = ROOT / "reports" / "figures" / "05_class_distribution.png"
    today = dt.date.today().strftime("%d %B %Y")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 18mm 16mm; }}
  body {{ font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1b1b1b; font-size: 11pt; line-height: 1.45; }}
  h1 {{ font-size: 20pt; margin: 0 0 2pt; }}
  .sub {{ color: #52514e; margin-bottom: 14pt; }}
  h2 {{ font-size: 13pt; margin: 16pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #d9d8d4; }}
  a {{ color: #2a78d6; }}
  .box {{ border: 1px solid #d9d8d4; border-radius: 6px; padding: 8pt 12pt; background: #fafaf8; }}
  .link {{ font-size: 13pt; font-weight: 600; word-break: break-all; }}
  pre {{ font-family: "SF Mono", Menlo, monospace; font-size: 8.6pt; line-height: 1.35; background: #fafaf8; border: 1px solid #d9d8d4; border-radius: 6px; padding: 8pt 10pt; white-space: pre; }}
  img {{ max-width: 100%; border: 1px solid #d9d8d4; border-radius: 6px; }}
  .cap {{ color: #52514e; font-size: 9.5pt; margin-top: 3pt; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 10pt; }}
  td, th {{ text-align: left; padding: 3pt 6pt; border-bottom: 1px solid #e6e5e1; vertical-align: top; }}
  th {{ color: #52514e; font-weight: 600; }}
  .pb {{ page-break-before: always; }}
</style></head><body>

<h1>Milestone 1 — Exploratory Data Analysis of MILK10k</h1>
<div class="sub">Computer Vision &amp; Speech Recognition &middot; Joshua Cole &middot; {today}</div>

<h2>1. Public GitHub repository</h2>
<div class="box">
  <div class="link"><a href="{REPO_URL}">{REPO_URL}</a></div>
  <div class="cap">EDA code: <code>notebooks/01_eda_milk10k.ipynb</code> (executed) &middot; helpers: <code>src/milk10k.py</code> &middot; figures: <code>reports/figures/</code></div>
</div>

<h2>2. Clinical task</h2>
<p>Classify a skin lesion into its diagnostic category from a <b>paired clinical close-up and dermatoscopic
image</b> plus patient metadata (age, sex, site, skin tone), so that malignant lesions (melanoma, basal cell
carcinoma, squamous cell carcinoma, &hellip;) are flagged for biopsy and benign ones are not over-treated.
MILK10k provides 5,240 lesions / 10,480 images with 11 top-level classes (48 fine diagnoses); ~96&nbsp;% of
labels are histopathology-confirmed.</p>

<h2>3. Project structure (code editor: VS Code)</h2>
<img src="{data_uri(screenshot)}" alt="VS Code screenshot of the project structure">
<div class="cap">Screenshot of the project in VS Code.</div>

<div class="pb"></div>
<pre>{html.escape(STRUCTURE)}</pre>

<h2>4. EDA summary</h2>
<table>
<tr><th>Finding</th><th>Implication for modelling</th></tr>
<tr><td>Exactly 1 clinical + 1 dermatoscopic image per lesion</td><td>Split by <code>lesion_id</code>; model the pair jointly</td></tr>
<tr><td><b>72&nbsp;% of lesions malignant; BCC alone 48&nbsp;%</b>, nevi only 14&nbsp;% — biopsy-enriched cohort</td><td>Use macro-F1 / balanced accuracy / per-class sensitivity, not accuracy</td></tr>
<tr><td>Five classes with &lt; 55 lesions (MAL_OTH has 9)</td><td>Stratified splits, class weighting / re-sampling</td></tr>
<tr><td>48 fine diagnoses, most with &lt; 10 lesions</td><td>Train at the 11-class level</td></tr>
<tr><td>Median age 65, 60&nbsp;% male, 61&nbsp;% skin-tone 3, &lt; 1&nbsp;% tones 0–1</td><td>Report metrics per tone / age band</td></tr>
<tr><td>Age, site and skin tone vary systematically by class</td><td>Legitimate priors but possible shortcuts — audit</td></tr>
<tr><td>Mixed resolutions; dermatoscopy brighter, with gel / hair / ink artefacts</td><td>Per-modality normalisation and augmentation</td></tr>
</table>
<p></p>
<img src="{data_uri(fig)}" alt="Lesions per top-level class">
<div class="cap">Lesions per top-level class (n = 5,240), coloured by malignancy. Full analysis in the notebook.</div>

</body></html>"""


def main() -> None:
    screenshot = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "editor_screenshot.png"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "milestone1_deliverable.pdf"
    if not screenshot.exists():
        sys.exit(f"screenshot not found: {screenshot}\nTake a screenshot of VS Code showing the project tree and save it there.")
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "deliverable.html"
        page.write_text(build_html(screenshot), encoding="utf-8")
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={out}", page.as_uri()],
            check=True, capture_output=True,
        )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
