# DocBerry

[![PyPI](https://img.shields.io/pypi/v/docberry?color=7B3FA0)](https://pypi.org/project/docberry/)
[![Python](https://img.shields.io/pypi/pyversions/docberry?color=34D399)](https://pypi.org/project/docberry/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<p align="center">
  <a href="https://aj-das-research.github.io/docberry/">
    <img src="https://img.shields.io/badge/%F0%9F%8C%90_Website-aj--das--research.github.io%2Fdocberry-7B3FA0?style=for-the-badge" alt="Website">
  </a>
</p>

Extract structured Markdown, tables, figures, and equations from scientific PDFs with proper reading order.

<p align="center">
  <img src="https://raw.githubusercontent.com/aj-das-research/docberry/main/docs/assets/image.png" alt="DocBerry Pipeline" width="800">
</p>

DocBerry wraps [Docling](https://github.com/DS4SD/docling) with a reading-order segmentation layer for two-column academic papers, multi-format asset extraction (PNG/PDF/SVG), and pluggable equation LaTeX enrichment backends.

---

## Features

- **Reading-order segmentation** — Detects full-width headers, two-column body text, and full-width figures/tables, then reorders them into natural reading order.
- **Structured Markdown output** — Tables replaced with image hyperlinks, equations with LaTeX blocks and image references.
- **Asset extraction** — Tables (PNG + CSV), figures (PNG + PDF + SVG), equations (PNG + PDF + SVG + LaTeX `.txt`).
- **Equation enrichment** — Choose from 4 backends: none, pix2tex (fast), Qwen3.5-0.8B VLM, or Docling CodeFormulaV2.
- **Auto-segment + convert** — One-shot pipeline that segments then converts.

---

## Installation

Requires **Python >= 3.10**.

```bash
# Core (segmentation + conversion + asset extraction)
pip install docberry

# With lightweight equation LaTeX OCR (pix2tex)
pip install 'docberry[pix2tex]'

# With Qwen3.5-0.8B VLM equation enrichment
pip install 'docberry[qwen]'

# With debug overlays (opencv)
pip install 'docberry[debug]'

# Everything
pip install 'docberry[all]'
```

### Pre-download model weights

Models auto-download on first use. To pre-download:

```bash
# Core Docling models only
docberry download-models

# All models (Docling + pix2tex + Qwen)
docberry download-models --all

# Specific extras
docberry download-models --pix2tex
docberry download-models --qwen
```

---

## Quick Start

### Python API

```python
from docberry import segment_pdf, convert_document

# Step 1 (optional): segment a two-column PDF into reading order
segment_pdf("paper.pdf", "paper_segmented.pdf")

# Step 2: convert to Markdown with asset extraction
result = convert_document(
    "paper_segmented.pdf",
    output_dir="output/",
    extract_assets=True,
    equation_enrichment="pix2tex",
)
print(result.markdown_path)   # output/paper_segmented.md
print(result.tables)          # number of tables extracted
print(result.figures)         # number of figures extracted
print(result.equations)       # number of equations extracted
print(result.elapsed_seconds) # total processing time

# One-shot: segment + convert in a single call
result = convert_document(
    "paper.pdf",
    output_dir="output/",
    extract_assets=True,
    auto_segment=True,
    equation_enrichment="pix2tex",
)
```

### CLI

```bash
# Full pipeline (segment + convert)
docberry convert paper.pdf -o output/ --extract-assets --auto-segment

# With equation enrichment
docberry convert paper.pdf -o output/ --extract-assets --equation-enrichment pix2tex

# Convert only (no segmentation)
docberry convert paper.pdf -o output/ --extract-assets

# Segment only
docberry segment paper.pdf -o paper_segmented.pdf

# Segment with debug overlays
docberry segment paper.pdf -o paper_segmented.pdf --debug-dir overlays/
```

---

## CLI Reference

### `docberry convert`

Convert a document to Markdown/JSON with optional asset extraction.

| Flag | Default | Description |
|------|---------|-------------|
| `source` (positional) | — | Path to a local file or URL |
| `--output-dir`, `-o` | `<stem>_output/` | Output directory |
| `--format` | `markdown` | Output format: `markdown` or `json` |
| `--extract-assets` | off | Extract tables, figures, equations |
| `--layout-model` | `heron` | `heron`, `egret-medium`, `egret-large`, `egret-xlarge` |
| `--pipeline` | `standard` | `standard` or `vlm` |
| `--equation-enrichment` | `none` | `none`, `pix2tex`, `qwen`, `docling` |
| `--auto-segment` | off | Segment PDF for reading order first |

### `docberry segment`

Segment a two-column PDF into reading-order pages.

| Flag | Default | Description |
|------|---------|-------------|
| `input` (positional) | — | Input PDF path |
| `--output`, `-o` | `segmented_output.pdf` | Output segmented PDF path |
| `--pages` | all | Page spec, 0-based (e.g. `0,2-4`) |
| `--debug-dir` | — | Directory for debug overlay images |

Layout tuning flags: `--line-merge-gap`, `--band-merge-gap`, `--block-padding`, `--segment-padding`, `--min-text-height`, `--line-split-gap-x`, `--num-bins`, `--single-coverage-threshold`, `--min-side-coverage`, `--min-center-gap-ratio`, `--min-band-height-ratio`, `--caption-merge-gap`, `--hrule-coverage`.

### `docberry download-models`

Pre-download model weights.

| Flag | Description |
|------|-------------|
| `--all` | Download all models (Docling + pix2tex + Qwen) |
| `--pix2tex` | Download pix2tex weights only |
| `--qwen` | Download Qwen3.5-0.8B weights only |

---

## Python API Reference

### `segment_pdf(input_pdf, output_pdf, page_spec, debug_dir, config)`

Segment a PDF into reading-order pages.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_pdf` | `str` | — | Path to source PDF |
| `output_pdf` | `str` | `"segmented_output.pdf"` | Output path |
| `page_spec` | `str \| None` | `None` | Page range, e.g. `"0,2-4"` |
| `debug_dir` | `str \| None` | `None` | Debug overlay directory |
| `config` | `LayoutConfig \| None` | `None` | Tuning parameters |

Returns: `list[Segment]`

### `convert_document(source, output_dir, ...)`

Convert a document to Markdown/JSON with optional asset extraction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | — | File path or URL |
| `output_dir` | `str \| None` | auto | Output directory |
| `output_format` | `str` | `"markdown"` | `"markdown"` or `"json"` |
| `extract_assets` | `bool` | `True` | Extract tables/figures/equations |
| `layout_model` | `str` | `"heron"` | Layout model name |
| `pipeline` | `str` | `"standard"` | `"standard"` or `"vlm"` |
| `equation_enrichment` | `str` | `"none"` | `"none"`, `"pix2tex"`, `"qwen"`, `"docling"` |
| `auto_segment` | `bool` | `False` | Run segmentation first |

Returns: `ConversionResult` with `.markdown_path`, `.tables`, `.figures`, `.equations`, `.elapsed_seconds`

---

## Equation Enrichment Comparison

| Method | Speed | Quality | Model Size | Extra Deps |
|--------|-------|---------|------------|------------|
| `none` | — | No LaTeX (images only) | — | — |
| `pix2tex` | Fast (~1s/eq) | Good for simple equations | ~100 MB | `pix2tex` |
| `qwen` | Medium (~3s/eq) | Good, handles complex notation | ~1.6 GB | `torch`, `transformers` |
| `docling` | Slow (~5s/eq) | High (CodeFormulaV2 VLM) | ~2 GB | Built into Docling |

---

## Output Structure

```
output/
  paper.md                    # Structured Markdown
  tables/
    table-1.png / .pdf / .svg
    table-1.csv
    table-1_caption.txt
  figures/
    figure-1.png / .pdf / .svg
    figure-1_caption.txt
  equations/
    equation-1.png / .pdf / .svg
    equation-1_latex.txt
```

---

## Acknowledgments

- [Docling](https://github.com/DS4SD/docling) — Core document conversion engine
- [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) — Lightweight LaTeX OCR
- [Qwen](https://github.com/QwenLM/Qwen) — Vision-language model for equation extraction
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF processing for layout segmentation

---

## License

MIT — see [LICENSE](LICENSE).

---

**[Website](https://aj-das-research.github.io/docberry/)** | **[PyPI](https://pypi.org/project/docberry/)** | **[GitHub](https://github.com/aj-das-research/docberry)**
