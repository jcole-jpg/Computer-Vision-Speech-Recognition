"""Export an executed notebook to PDF: nbconvert -> HTML -> headless Chrome -> PDF.

    python scripts/export_notebook_pdf.py homework_part_a.ipynb

This is the homework's sanctioned fallback ("export to HTML and use Print to PDF")
because no LaTeX toolchain is installed. Printing with headless Chrome instead of
clicking through a browser keeps the export reproducible from the command line.

The CSS below forces page breaks not to fall inside an output block, which is what
otherwise cuts plots in half at the page edge.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

#: Injected before </head> in the nbconvert HTML.
PRINT_CSS = """
<style>
  @page { size: A4; margin: 12mm 10mm; }
  body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  /* never split a cell, a figure or a table across a page edge */
  .jp-Cell, .jp-OutputArea-child, .jp-RenderedImage, figure, table, pre {
      break-inside: avoid; page-break-inside: avoid;
  }
  /* start each exercise on a fresh page so sections stay in order and legible */
  h2 { break-before: page; page-break-before: always; margin-top: 0; }
  h1 + h2, h2:first-of-type { break-before: auto; page-break-before: auto; }
  img, svg { max-width: 100% !important; height: auto !important; }

  /* Wrap long source lines. nbconvert puts code in `.highlight pre` inside a
     `table-layout: fixed; overflow: hidden` container, which CLIPS anything wider
     than the page instead of wrapping it - so both must be overridden or the
     right-hand end of long lines is silently missing from the PDF. */
  .jp-InputArea, .jp-InputArea-editor, .jp-CodeMirrorEditor, .cm-editor, .highlight {
      overflow: visible !important;
  }
  .highlight pre, .jp-InputArea-editor pre {
      white-space: pre-wrap !important;
      word-break: break-word !important;
      overflow-wrap: anywhere !important;
      font-size: 9.5px !important; line-height: 1.35 !important;
  }
  .jp-OutputArea-output pre {
      white-space: pre-wrap !important; word-break: break-word !important;
      font-size: 9px !important; line-height: 1.3 !important;
  }
  #notebook-container, .jp-Notebook { padding: 0 !important; }
</style>
"""


def to_html(nb_path: Path) -> Path:
    html_path = nb_path.with_suffix(".html")
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "html", "--log-level", "WARN",
         str(nb_path), "--output", html_path.name, "--output-dir", str(nb_path.parent)],
        check=True, cwd=ROOT,
    )
    html = html_path.read_text()
    if "</head>" not in html:
        raise RuntimeError("unexpected nbconvert output: no </head>")
    html_path.write_text(html.replace("</head>", PRINT_CSS + "</head>", 1))
    return html_path


def to_pdf(html_path: Path, pdf_path: Path) -> Path:
    if not CHROME.is_file():
        raise SystemExit(
            f"Google Chrome not found at {CHROME}.\n"
            "Open the generated .html and use the browser's Print > Save as PDF."
        )
    subprocess.run(
        [str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=20000",
         f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
        check=True, capture_output=True,
    )
    return pdf_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", nargs="?", default="homework_part_a.ipynb")
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    nb_path = (ROOT / args.notebook).resolve()
    if not nb_path.is_file():
        raise SystemExit(f"notebook not found: {nb_path}")

    pdf_path = nb_path.with_suffix(".pdf")
    html_path = to_html(nb_path)
    to_pdf(html_path, pdf_path)
    if not args.keep_html:
        html_path.unlink(missing_ok=True)

    size_mb = pdf_path.stat().st_size / 1024**2
    print(f"wrote {pdf_path.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
