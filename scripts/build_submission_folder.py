"""Assemble everything the teacher needs into Finished/, plus one zip to email.

    python scripts/build_submission_folder.py

Collects the graded deliverables from around the repo into a single, numbered
folder, renders the Markdown reports to PDF so they read without a Markdown
viewer, and writes one .zip that fits comfortably in an email.

Figures are re-encoded for the email copy: the photo-heavy galleries go to JPEG
(8.2 MB -> ~1.2 MB, visually identical at this size) while charts stay PNG so
their text stays sharp. Full-resolution originals remain in reports/figures/ and
in the repo, and START_HERE says so.

Nothing here is a source of truth: every file is copied from where it is
generated, so re-running build_pipeline.py then this script is always consistent.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import mistune
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FINISHED = ROOT / "Finished"
ZIP_PATH = ROOT / "Cole_Joshua_CVSR_Sessions_1-3.zip"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

REPO_URL = "https://github.com/jcole-jpg/Computer-Vision-Speech-Recognition"
STUDENT = "Joshua Cole"

#: Figures that are photographs -> JPEG. Charts keep their lossless PNG.
PHOTO_FIGURES = {"b2_class_gallery.png", "b7_augmentations.png", "b8_train_batch.png"}

CSS = """
<style>
  @page { size: A4; margin: 16mm 14mm; }
  body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; max-width: 100%; }
  h1 { font-size: 19pt; margin: 0 0 .3em; border-bottom: 2px solid #2a78d6; padding-bottom: .2em; }
  h2 { font-size: 13pt; margin: 1.4em 0 .4em; color: #2a78d6; break-after: avoid; }
  h3 { font-size: 11pt; margin: 1em 0 .3em; }
  table { border-collapse: collapse; width: 100%; margin: .7em 0; font-size: 9pt;
          break-inside: avoid; }
  th, td { border: 1px solid #ccc; padding: 4px 7px; text-align: left; vertical-align: top; }
  th { background: #eef4fb; }
  code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
  pre { background: #f7f7f7; padding: 8px 10px; border-radius: 4px; overflow-x: auto;
        font-size: 8.5pt; break-inside: avoid; }
  pre code { background: none; }
  blockquote { border-left: 3px solid #2a78d6; margin: .7em 0; padding: .2em 0 .2em 1em;
               color: #444; background: #f8fafd; }
  a { color: #2a78d6; }
  hr { border: none; border-top: 1px solid #ddd; margin: 1.4em 0; }
  img { max-width: 100%; }
</style>
"""


#: Denser profile for the Milestone 1 report, which has a hard 2-page limit (B9).
#: Only type size and spacing change - no content is hidden or shrunk below
#: comfortable reading size (9pt body, 8pt tables).
COMPACT_CSS = """
<style>
  @page { size: A4; margin: 11mm 11mm; }
  body { font-size: 9pt; line-height: 1.32; }
  h1 { font-size: 15pt; margin: 0 0 .2em; }
  h2 { font-size: 10.5pt; margin: .75em 0 .25em; }
  p { margin: .4em 0; }
  ul { margin: .35em 0; padding-left: 1.1em; }
  li { margin: .18em 0; }
  table { font-size: 7.8pt; margin: .45em 0; }
  th, td { padding: 2px 5px; }
  hr { display: none; }
</style>
"""


def md_to_pdf(md_path: Path, pdf_path: Path, title: str = "",
              compact: bool = False) -> Path | None:
    """Render a Markdown file to PDF via headless Chrome (no LaTeX needed)."""
    html_body = mistune.html(md_path.read_text())
    style = CSS + (COMPACT_CSS if compact else "")
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title or md_path.stem}</title>{style}</head>"
            f"<body>{html_body}</body></html>")
    tmp = pdf_path.with_suffix(".tmp.html")
    tmp.write_text(html)
    try:
        if not CHROME.is_file():
            print(f"    ! Chrome not found; left {tmp.name} for manual printing")
            return None
        subprocess.run(
            [str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             "--virtual-time-budget=10000",
             f"--print-to-pdf={pdf_path}", tmp.resolve().as_uri()],
            check=True, capture_output=True,
        )
        return pdf_path
    finally:
        tmp.unlink(missing_ok=True)


def copy_figure(src: Path, dest_dir: Path) -> Path:
    """Copy a figure, re-encoding the photo-heavy ones to JPEG to fit an email."""
    if src.name in PHOTO_FIGURES:
        img = Image.open(src).convert("RGB")
        w, h = img.size
        scale = min(1.0, 1800 / w)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        out = dest_dir / (src.stem + ".jpg")
        img.save(out, "JPEG", quality=88, optimize=True)
        return out
    out = dest_dir / src.name
    shutil.copy2(src, out)
    return out


def human(n: int) -> str:
    return f"{n / 1024**2:.1f} MB" if n >= 1024**2 else f"{n / 1024:.0f} KB"


def tree_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def build_start_here() -> str:
    return f"""# Homework Sessions 1-3 + Course Project Milestone 1

**Student:** {STUDENT}
**Course:** Computer Vision and Speech Recognition - EADA
**Repository:** <{REPO_URL}>
**Assembled:** {date.today().isoformat()}

---

## Read these two first

| # | File | What it is |
|---|---|---|
| 1 | `01_Part_A_notebook/homework_part_a.pdf` | **Part A** - all 12 exercises (A1.1-A3.6) with code, outputs and written answers |
| 2 | `02_Part_B_milestone1/milestone1_report.pdf` | **Part B** - the Milestone 1 report (label strategy, splits, imbalance, preprocessing, risks) |

`SUBMISSION.md` is the required link list: every deliverable with its path in the repo.

---

## What is in this folder

```
Finished/
├── 00_START_HERE.pdf / .md          this page
├── SUBMISSION.md                    the required link list
│
├── 01_Part_A_notebook/
│   ├── homework_part_a.pdf          the graded notebook, 33 pages
│   └── homework_part_a.ipynb        the notebook itself (runs top to bottom)
│
├── 02_Part_B_milestone1/
│   ├── milestone1_report.pdf/.md    the Milestone 1 report, 2 pages (B9)
│   ├── prevalence_analysis.pdf/.md  why MILK10k is not real-world prevalence (B2)
│   ├── README.pdf/.md               project README: setup, structure, decisions (B0)
│   ├── artifacts/                   splits, label map, class weights, norm stats,
│   │                                data-quality report, Session-2 re-check
│   ├── figures/                     the B2 / B7 / B8 figures
│   └── source_code/                 src/milk10k (the package), scripts/, tests/
│
└── 03_earlier_sessions/             exercises 1-3 from Sessions 1 and 2
```

---

## Where each requirement is answered

### Part A - all in `01_Part_A_notebook/homework_part_a.pdf`

| Exercise | Section |
|---|---|
| A1.1 Cross-check the two label files | `## A1.1` |
| A1.2 Five questions in pandas | `## A1.2` |
| A1.3 Lesion-level table (5,240 rows) | `## A1.3` |
| A1.4 Task framing | `## A1.4` |
| A2.1 Arrays, views, memory budget | `## A2.1` |
| A2.2 JPEG compression + file-size shortcut | `## A2.2` |
| A3.1 Measuring the leak | `## A3.1` |
| A3.2 Three-way split + property test | `## A3.2` |
| A3.3 Why StratifiedGroupKFold | `## A3.3` |
| A3.4 Metrics and the cost of errors | `## A3.4` |
| A3.5 Augmentation audit | `## A3.5` |
| A3.6 Lesion-level Dataset | `## A3.6` |

### Part B

| B | Requirement | Where |
|---|---|---|
| B0 | README: task, setup, structure, data rules, decisions | `02_.../README.pdf` |
| B1 | Full-dataset integrity check | `02_.../artifacts/image_size_summary.csv` |
| B2 | Class distribution, 11-class log scale, mapping | `02_.../figures/b2_class_distribution.png` |
| B2 | 3x4 gallery, one example per class | `02_.../figures/b2_class_gallery.jpg` |
| B2 | Session 2 findings re-tested on the full dataset | `02_.../artifacts/session2_findings_recheck.md` |
| B2 | Why MILK10k does not reflect real prevalence | `02_.../prevalence_analysis.pdf` |
| B3 | Label strategy + label map | `02_.../artifacts/label_map.json` |
| B4 | Data-quality report | `02_.../artifacts/data_quality_report.md` |
| B5 | Train / val / test splits | `02_.../artifacts/splits/*.csv` |
| B5 | Split verification | `02_.../artifacts/split_verification.csv` |
| B6 | Preprocessing decisions | report, section 5 |
| B7 | Transforms in an importable module | `02_.../source_code/src/milk10k/augment.py` |
| B7 | 1 original + 7 augmentations, 3 classes | `02_.../figures/b7_augmentations.jpg` |
| B7 | Augmentation table with justification | `02_.../artifacts/augmentation_table.csv` |
| B8 | Dataset / DataLoaders / sampler | `02_.../source_code/src/milk10k/datasets.py` |
| B8 | Class weights, normalisation stats (train only) | `02_.../artifacts/class_weights.json`, `norm_stats.json` |
| B8 | Batch of 16 after transforms | `02_.../figures/b8_train_batch.jpg` |
| B9 | Milestone 1 report | `02_.../milestone1_report.pdf` |

---

## Reproducing everything

The raw MILK10k images are not included (~350 MB, CC-BY-NC forbids redistribution).
With the dataset in `data/raw/milk10k/` (or `$MILK10K_DATA` pointing at it):

```bash
pip install -r requirements.txt
python scripts/build_pipeline.py             # every Part B artefact, ~35 s, deterministic
python scripts/build_homework_notebook.py --run
python scripts/export_notebook_pdf.py homework_part_a.ipynb
python -m pytest tests/ -q                   # 102 tests
```

The splits are byte-identical on every rebuild (seed 42, created 2026-09-24).

> **Note on the figures.** The photo galleries in `figures/` are JPEG copies, resized
> to keep this folder small enough to email. The full-resolution PNGs are in the
> repository under `reports/figures/`.
"""


def main() -> int:
    if not (ROOT / "homework_part_a.pdf").is_file():
        raise SystemExit("homework_part_a.pdf is missing - run the notebook export first")

    if FINISHED.exists():
        shutil.rmtree(FINISHED)
    part_a = FINISHED / "01_Part_A_notebook"
    part_b = FINISHED / "02_Part_B_milestone1"
    earlier = FINISHED / "03_earlier_sessions"
    for d in (part_a, part_b, earlier, part_b / "figures", part_b / "source_code"):
        d.mkdir(parents=True, exist_ok=True)

    print("Part A")
    for name in ("homework_part_a.pdf", "homework_part_a.ipynb"):
        shutil.copy2(ROOT / name, part_a / name)
        print(f"  {name}  {human((part_a / name).stat().st_size)}")

    print("Part B")
    for name in ("milestone1_report.md", "prevalence_analysis.md"):
        shutil.copy2(ROOT / "reports" / name, part_b / name)
    shutil.copy2(ROOT / "README.md", part_b / "README.md")
    # compact=True for the report: B9 caps it at 2 pages.
    md_to_pdf(part_b / "milestone1_report.md", part_b / "milestone1_report.pdf",
              "Milestone 1 Report", compact=True)
    md_to_pdf(part_b / "prevalence_analysis.md", part_b / "prevalence_analysis.pdf",
              "Why MILK10k does not reflect real prevalence")
    md_to_pdf(part_b / "README.md", part_b / "README.pdf", "Project README")
    for f in ("milestone1_report.pdf", "prevalence_analysis.pdf", "README.pdf"):
        print(f"  {f:24s} {human((part_b / f).stat().st_size)}")

    shutil.copytree(ROOT / "artifacts", part_b / "artifacts", dirs_exist_ok=True)
    print(f"  artifacts/             {human(tree_size(part_b / 'artifacts'))}")

    for fig in sorted((ROOT / "reports" / "figures").glob("b*.png")):
        out = copy_figure(fig, part_b / "figures")
        print(f"  figures/{out.name:28s} {human(out.stat().st_size)}")

    for src, dst in [(ROOT / "src", "src"), (ROOT / "scripts", "scripts"),
                     (ROOT / "tests", "tests")]:
        shutil.copytree(src, part_b / "source_code" / dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                      ".pytest_cache", ".ipynb_checkpoints"))
    shutil.copy2(ROOT / "requirements.txt", part_b / "source_code" / "requirements.txt")
    print(f"  source_code/           {human(tree_size(part_b / 'source_code'))}")

    print("Earlier sessions")
    for pdf in sorted((ROOT / "exercises").glob("*.pdf")):
        shutil.copy2(pdf, earlier / pdf.name)
        print(f"  {pdf.name}  {human(pdf.stat().st_size)}")

    print("Top level")
    shutil.copy2(ROOT / "SUBMISSION.md", FINISHED / "SUBMISSION.md")
    start = FINISHED / "00_START_HERE.md"
    start.write_text(build_start_here())
    md_to_pdf(start, FINISHED / "00_START_HERE.pdf", "START HERE")
    print("  00_START_HERE.pdf / .md, SUBMISSION.md")

    # one attachment
    ZIP_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(FINISHED.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                z.write(f, Path("Finished") / f.relative_to(FINISHED))

    total = tree_size(FINISHED)
    zsize = ZIP_PATH.stat().st_size
    print(f"\nFinished/  {human(total)} in {sum(1 for f in FINISHED.rglob('*') if f.is_file())} files")
    print(f"{ZIP_PATH.name}  {human(zsize)}")
    print("OK for a 25 MB email limit" if zsize < 20 * 1024**2
          else "WARNING: close to the usual 25 MB email limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
