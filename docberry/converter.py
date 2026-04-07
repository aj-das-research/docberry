"""
Document conversion using Docling — converts PDF/DOCX/HTML/images to
Markdown or JSON with structured asset extraction.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling_core.types.doc.document import FormulaItem, TextItem
from docling_core.types.doc.base import BoundingBox, CoordOrigin

from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_EGRET_MEDIUM,
    DOCLING_LAYOUT_EGRET_XLARGE,
    DOCLING_LAYOUT_HERON,
)
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    PdfPipelineOptions,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 900 / 72
IMAGE_DPI = 900

LAYOUT_MODELS = {
    "heron": DOCLING_LAYOUT_HERON,
    "egret-medium": DOCLING_LAYOUT_EGRET_MEDIUM,
    "egret-large": DOCLING_LAYOUT_EGRET_LARGE,
    "egret-xlarge": DOCLING_LAYOUT_EGRET_XLARGE,
}

ADJACENT_GAP_FRACTION = 0.05


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConversionResult:
    """Result returned by :func:`convert_document`."""

    markdown_path: Optional[Path] = None
    json_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    tables: int = 0
    figures: int = 0
    equations: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _AssetRecord:
    element: object
    kind: str  # "table" or "figure"
    caption: str
    page_no: int
    bbox: Optional[BoundingBox]
    merged_bboxes: list[BoundingBox] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Asset collection and post-processing
# ---------------------------------------------------------------------------


_CAPTION_PATTERN = re.compile(
    r"^\s*(Figure|Table|Fig\.)\s+(S?\d+[a-z]?)\b",
    re.IGNORECASE,
)

_TABLE_CAPTION_RE = re.compile(r"^\s*table\s", re.IGNORECASE)


@dataclass
class _CaptionCandidate:
    kind: str
    label_id: str
    text: str
    page_no: int
    iter_idx: int


def _collect_assets(doc) -> list[_AssetRecord]:
    assets: list[_AssetRecord] = []
    for element, _level in doc.iterate_items():
        if isinstance(element, (TableItem, PictureItem)):
            kind = "table" if isinstance(element, TableItem) else "figure"
            caption = element.caption_text(doc)
            page_no = element.prov[0].page_no if element.prov else -1
            bbox = element.prov[0].bbox if element.prov else None
            assets.append(_AssetRecord(
                element=element, kind=kind, caption=caption,
                page_no=page_no, bbox=bbox,
            ))
    return assets


def _recover_missing_captions(assets: list[_AssetRecord], doc) -> list[_AssetRecord]:
    candidates: list[_CaptionCandidate] = []
    for idx, (element, _level) in enumerate(doc.iterate_items()):
        if not isinstance(element, TextItem):
            continue
        m = _CAPTION_PATTERN.match(element.text)
        if not m:
            continue
        text = element.text.strip()
        kind_word = m.group(1).lower()
        kind = "table" if kind_word == "table" else "figure"
        label_id = m.group(2)
        page_no = element.prov[0].page_no if element.prov else -1
        candidates.append(_CaptionCandidate(
            kind=kind, label_id=label_id, text=text,
            page_no=page_no, iter_idx=idx,
        ))

    if not candidates:
        return assets

    used_candidates: set[int] = set()
    for asset in assets:
        if asset.caption.strip():
            continue

        best: Optional[_CaptionCandidate] = None
        best_dist = float("inf")
        best_idx = -1
        for ci, cand in enumerate(candidates):
            if ci in used_candidates:
                continue
            if cand.kind != asset.kind:
                continue
            page_dist = abs(cand.page_no - asset.page_no)
            if page_dist > 1:
                continue
            if page_dist < best_dist:
                best_dist = page_dist
                best = cand
                best_idx = ci

        if best is not None:
            asset.caption = best.text
            used_candidates.add(best_idx)
            log.info(
                "Recovered caption for %s on page %d: '%s'",
                asset.kind, asset.page_no, best.text[:80],
            )

    return assets


def _reclassify_tables(assets: list[_AssetRecord]) -> list[_AssetRecord]:
    for asset in assets:
        if asset.kind == "figure" and _TABLE_CAPTION_RE.match(asset.caption):
            log.info("Reclassifying figure as table (caption: '%s')", asset.caption[:80])
            asset.kind = "table"
    return assets


def _min_gap(a: BoundingBox, b: BoundingBox) -> float:
    if a.coord_origin == CoordOrigin.TOPLEFT:
        v_gap = max(0.0, max(a.t, b.t) - min(a.b, b.b))
        if a.b <= b.t or b.b <= a.t:
            v_gap = abs(b.t - a.b) if a.t <= b.t else abs(a.t - b.b)
    else:
        v_gap = max(0.0, max(a.b, b.b) - min(a.t, b.t))
        if a.t <= b.b or b.t <= a.b:
            v_gap = abs(a.b - b.t) if a.t >= b.t else abs(b.b - a.t)

    h_gap = max(0.0, max(a.l, b.l) - min(a.r, b.r))
    return min(v_gap, h_gap)


def _captions_match(cap_a: str, cap_b: str) -> bool:
    a, b = cap_a.strip(), cap_b.strip()
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def _should_merge_figures(group: list[_AssetRecord], candidate: _AssetRecord, doc) -> bool:
    page = doc.pages.get(group[0].page_no)
    page_height = page.size.height if page and page.size else 1000.0

    any_caption = any(a.caption.strip() for a in group) or candidate.caption.strip()
    gap = _min_gap(group[-1].bbox, candidate.bbox)

    if _captions_match(group[0].caption, candidate.caption):
        return True
    if gap <= page_height * ADJACENT_GAP_FRACTION:
        return True
    no_caption_on_one_side = (
        not candidate.caption.strip() or not group[0].caption.strip()
    )
    if no_caption_on_one_side and any_caption and gap <= page_height * 0.10:
        return True
    return False


def _merge_adjacent_figures(assets: list[_AssetRecord], doc) -> list[_AssetRecord]:
    if not assets:
        return assets
    merged: list[_AssetRecord] = []
    i = 0
    while i < len(assets):
        current = assets[i]
        if current.kind != "figure" or current.bbox is None:
            merged.append(current)
            i += 1
            continue

        group = [current]
        j = i + 1
        while j < len(assets):
            nxt = assets[j]
            if nxt.kind != "figure" or nxt.bbox is None:
                break
            if nxt.page_no != current.page_no:
                break
            if not _should_merge_figures(group, nxt, doc):
                break
            group.append(nxt)
            j += 1

        if len(group) == 1:
            merged.append(current)
        else:
            all_bboxes = [a.bbox for a in group if a.bbox is not None]
            enclosing = BoundingBox.enclosing_bbox(all_bboxes)
            captions = list(dict.fromkeys(
                a.caption for a in group if a.caption.strip()
            ))
            log.info(
                "Merging %d figure fragments on page %d into one figure",
                len(group), current.page_no,
            )
            merged.append(_AssetRecord(
                element=group[0].element,
                kind="figure",
                caption=" ".join(captions),
                page_no=current.page_no,
                bbox=enclosing,
                merged_bboxes=all_bboxes,
            ))
        i = j
    return merged


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _crop_from_page(
    bbox: BoundingBox, page_no: int, doc,
) -> Optional[PILImage.Image]:
    page = doc.pages.get(page_no)
    if page is None or page.size is None or page.image is None:
        return None
    page_image = page.image.pil_image
    if page_image is None:
        return None

    crop_bbox = (
        bbox.to_top_left_origin(page_height=page.size.height)
        .scale_to_size(old_size=page.size, new_size=page.image.size)
    )
    return page_image.crop(crop_bbox.as_tuple())


def _save_image_multi_format(img: PILImage.Image, base_path: Path) -> bool:
    """Save an image as PNG, PDF, and SVG at *base_path* (no extension)."""
    png_path = base_path.with_suffix(".png")
    with png_path.open("wb") as fp:
        img.save(fp, "PNG", dpi=(IMAGE_DPI, IMAGE_DPI))
    log.info("Saved %s", png_path)

    pdf_path = base_path.with_suffix(".pdf")
    rgb_img = img.convert("RGB") if img.mode == "RGBA" else img
    rgb_img.save(str(pdf_path), "PDF", resolution=IMAGE_DPI)
    log.info("Saved %s", pdf_path)

    svg_path = base_path.with_suffix(".svg")
    w_in = img.width / IMAGE_DPI
    h_in = img.height / IMAGE_DPI
    buf = BytesIO()
    img.save(buf, "PNG", dpi=(IMAGE_DPI, IMAGE_DPI))
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_in:.4f}in" height="{h_in:.4f}in" '
        f'viewBox="0 0 {img.width} {img.height}">\n'
        f'  <image width="{img.width}" height="{img.height}" '
        f'href="data:image/png;base64,{b64}"/>\n'
        f'</svg>\n'
    )
    svg_path.write_text(svg_content, encoding="utf-8")
    log.info("Saved %s", svg_path)
    return True


def _get_asset_image(asset: _AssetRecord, doc) -> Optional[PILImage.Image]:
    if asset.merged_bboxes:
        return _crop_from_page(asset.bbox, asset.page_no, doc)
    return asset.element.get_image(doc)


def _save_asset_image(asset: _AssetRecord, doc, base_path: Path) -> bool:
    img = _get_asset_image(asset, doc)
    if img is None:
        return False
    return _save_image_multi_format(img, base_path)


# ---------------------------------------------------------------------------
# Markdown post-processing
# ---------------------------------------------------------------------------


def _replace_tables_with_images(
    md_path: Path, table_count: int, tables_dir: Path,
) -> None:
    if table_count == 0:
        return

    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    result: list[str] = []
    table_idx = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            block_start = i
            while i < len(lines) and lines[i].startswith("|"):
                i += 1
            block = lines[block_start:i]
            has_separator = any(
                all(c in "-| " for c in row) and "---" in row
                for row in block
            )
            if has_separator:
                table_idx += 1
                img_path = tables_dir / f"table-{table_idx}.png"
                if img_path.exists():
                    rel_path = img_path.relative_to(md_path.parent)
                    cap_path = tables_dir / f"table-{table_idx}_caption.txt"
                    alt = f"Table {table_idx}"
                    if cap_path.exists():
                        cap_text = cap_path.read_text(encoding="utf-8").strip()
                        first_sentence = cap_text.split(".")[0].strip()
                        if first_sentence:
                            alt = first_sentence
                    result.append(f"![{alt}]({rel_path})")
                    result.append("")
                    log.info("Replaced Markdown table %d with image ref: %s", table_idx, rel_path)
                else:
                    result.extend(block)
            else:
                result.extend(block)
        else:
            result.append(line)
            i += 1

    md_path.write_text("\n".join(result), encoding="utf-8")
    log.info("Post-processed %s: replaced %d/%d tables with images", md_path, table_idx, table_count)


def _replace_formula_placeholders(
    md_path: Path, eq_latex_map: dict[int, str],
) -> None:
    if not eq_latex_map:
        return

    _FORMULA_PLACEHOLDER = "<!-- formula-not-decoded -->"
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    result: list[str] = []
    eq_idx = 0

    for line in lines:
        if line.strip() == _FORMULA_PLACEHOLDER:
            eq_idx += 1
            if eq_idx in eq_latex_map:
                result.append(f"$${eq_latex_map[eq_idx]}$$")
                log.info("Replaced formula placeholder %d with enriched LaTeX", eq_idx)
            else:
                result.append(line)
        else:
            result.append(line)

    md_path.write_text("\n".join(result), encoding="utf-8")
    log.info("Post-processed %s: replaced %d/%d formula placeholders", md_path, len(eq_latex_map), eq_idx)


def _add_equation_image_refs(
    md_path: Path, eq_count: int, equations_dir: Path,
) -> None:
    if eq_count == 0:
        return

    _FORMULA_PLACEHOLDER = "<!-- formula-not-decoded -->"
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    result: list[str] = []
    eq_idx = 0
    i = 0

    def _append_eq_image(idx: int) -> None:
        img_path = equations_dir / f"equation-{idx}.png"
        if img_path.exists():
            rel_path = img_path.relative_to(md_path.parent)
            result.append("")
            result.append(f"![Equation {idx}]({rel_path})")
            log.info("Added image ref for equation %d: %s", idx, rel_path)

    while i < len(lines):
        line = lines[i]
        if line.strip() == _FORMULA_PLACEHOLDER:
            eq_idx += 1
            result.append(line)
            _append_eq_image(eq_idx)
            i += 1
        elif line.startswith("$$") and line.rstrip().endswith("$$") and len(line.strip()) > 4:
            eq_idx += 1
            result.append(line)
            _append_eq_image(eq_idx)
            i += 1
        elif line.startswith("$$") and not line.rstrip().endswith("$$"):
            block_start = i
            i += 1
            while i < len(lines) and not lines[i].rstrip().endswith("$$"):
                i += 1
            if i < len(lines):
                i += 1
            block = lines[block_start:i]
            eq_idx += 1
            result.extend(block)
            _append_eq_image(eq_idx)
        else:
            result.append(line)
            i += 1

    md_path.write_text("\n".join(result), encoding="utf-8")
    log.info("Post-processed %s: added image refs for %d/%d equations", md_path, eq_idx, eq_count)


# ---------------------------------------------------------------------------
# Equation extraction
# ---------------------------------------------------------------------------


def _extract_equations(
    doc, equations_dir: Path, enrich_method: str = "none",
) -> tuple[int, dict[int, str]]:
    """Collect FormulaItem elements, save images, and optionally OCR them.

    *enrich_method*: ``"none"`` | ``"pix2tex"`` | ``"qwen"`` | ``"docling"``.
    """
    from docberry._enrichment import init_pix2tex, init_qwen_vlm, qwen_image_to_latex

    equations_dir.mkdir(parents=True, exist_ok=True)
    eq_count = 0
    eq_latex_map: dict[int, str] = {}

    pix2tex_model = None
    qwen_model = None
    qwen_processor = None

    if enrich_method == "pix2tex":
        pix2tex_model = init_pix2tex()
    elif enrich_method == "qwen":
        qwen_model, qwen_processor = init_qwen_vlm()

    for element, _level in doc.iterate_items():
        if not isinstance(element, FormulaItem):
            continue
        img = element.get_image(doc)
        if img is None:
            log.warning(
                "Could not get image for equation on page %s",
                element.prov[0].page_no if element.prov else "?",
            )
            continue
        eq_count += 1
        base = equations_dir / f"equation-{eq_count}"
        _save_image_multi_format(img, base)

        latex_text = element.text
        if pix2tex_model is not None:
            try:
                latex_text = pix2tex_model(img)
                eq_latex_map[eq_count] = latex_text
                log.info("pix2tex equation %d: %s", eq_count, latex_text[:120])
            except Exception as exc:
                log.warning("pix2tex failed for equation %d: %s", eq_count, exc)
        elif qwen_model is not None:
            try:
                latex_text = qwen_image_to_latex(img, qwen_model, qwen_processor)
                eq_latex_map[eq_count] = latex_text
                log.info("Qwen VLM equation %d: %s", eq_count, latex_text[:120])
            except Exception as exc:
                log.warning("Qwen VLM failed for equation %d: %s", eq_count, exc)

        latex_path = equations_dir / f"equation-{eq_count}_latex.txt"
        latex_path.write_text(latex_text, encoding="utf-8")
        log.info("Saved %s", latex_path)

    log.info("Extracted %d equation images", eq_count)
    return eq_count, eq_latex_map


# ---------------------------------------------------------------------------
# Docling converter builder
# ---------------------------------------------------------------------------


def _build_converter(
    extract_assets: bool,
    layout_model: str = "heron",
    pipeline: str = "standard",
    enrich_formulas: bool = False,
) -> DocumentConverter:
    if pipeline == "vlm":
        from docling.datamodel import vlm_model_specs

        vlm_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.SMOLDOCLING_MLX,
            generate_page_images=True,
            generate_picture_images=extract_assets,
        )
        log.info("Using VLM pipeline (SmolDocling MLX)")
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=vlm_options,
                )
            }
        )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.layout_options = LayoutOptions(
        model_spec=LAYOUT_MODELS[layout_model]
    )
    log.info("Using standard pipeline with layout model: %s", layout_model)

    if extract_assets:
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True

    if enrich_formulas:
        from docling.datamodel.pipeline_options import CodeFormulaVlmOptions

        pipeline_options.do_formula_enrichment = True
        pipeline_options.code_formula_options = CodeFormulaVlmOptions.from_preset(
            "codeformulav2"
        )
        log.info("Formula enrichment enabled (CodeFormulaV2)")

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# ---------------------------------------------------------------------------
# Internal extraction driver
# ---------------------------------------------------------------------------


def _extract_assets(conv_res, output_dir: Path, enrich_method: str = "none") -> dict:
    import pandas as pd

    doc = conv_res.document
    doc_stem = conv_res.input.file.stem

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    assets = _collect_assets(doc)
    assets = _recover_missing_captions(assets, doc)
    assets = _reclassify_tables(assets)
    assets = _merge_adjacent_figures(assets, doc)

    table_count = 0
    figure_count = 0

    for asset in assets:
        if asset.kind == "table":
            table_count += 1
            prefix = f"table-{table_count}"
            _save_asset_image(asset, doc, tables_dir / prefix)
            if isinstance(asset.element, TableItem):
                try:
                    df: pd.DataFrame = asset.element.export_to_dataframe(doc=doc)
                    csv_path = tables_dir / f"{prefix}.csv"
                    df.to_csv(csv_path, index=False)
                    log.info("Saved %s", csv_path)
                except Exception as exc:
                    log.warning("Could not export table %d to CSV: %s", table_count, exc)
            if asset.caption.strip():
                cap_path = tables_dir / f"{prefix}_caption.txt"
                cap_path.write_text(asset.caption, encoding="utf-8")
                log.info("Saved %s", cap_path)
        else:
            figure_count += 1
            prefix = f"figure-{figure_count}"
            _save_asset_image(asset, doc, figures_dir / prefix)
            if asset.caption.strip():
                cap_path = figures_dir / f"{prefix}_caption.txt"
                cap_path.write_text(asset.caption, encoding="utf-8")
                log.info("Saved %s", cap_path)

    equations_dir = output_dir / "equations"
    eq_count, eq_latex_map = _extract_equations(doc, equations_dir, enrich_method)

    md_path = output_dir / f"{doc_stem}.md"
    doc.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)
    log.info("Saved Markdown with image references: %s", md_path)

    _replace_tables_with_images(md_path, table_count, tables_dir)
    _replace_formula_placeholders(md_path, eq_latex_map)
    _add_equation_image_refs(md_path, eq_count, equations_dir)

    return {"tables": table_count, "figures": figure_count, "equations": eq_count, "md_path": md_path}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_document(
    source: str,
    output_dir: Optional[str] = None,
    output_format: str = "markdown",
    extract_assets: bool = True,
    layout_model: str = "heron",
    pipeline: str = "standard",
    equation_enrichment: str = "none",
    auto_segment: bool = False,
) -> ConversionResult:
    """Convert a document to Markdown (or JSON) with optional asset extraction.

    This is the main public API for DocBerry.

    Args:
        source: Path to a local file or a URL.
        output_dir: Directory for output files. Derived from *source* if
            ``None``.
        output_format: ``"markdown"`` or ``"json"``.
        extract_assets: If ``True``, extract tables, figures, and equations
            as separate image files and post-process the Markdown.
        layout_model: Docling layout model: ``"heron"``, ``"egret-medium"``,
            ``"egret-large"``, or ``"egret-xlarge"``.
        pipeline: ``"standard"`` or ``"vlm"``.
        equation_enrichment: LaTeX extraction method for equations.
            ``"none"`` | ``"pix2tex"`` | ``"qwen"`` | ``"docling"``
            (docling uses CodeFormulaV2 VLM during Docling conversion).
        auto_segment: If ``True``, run :func:`segment_pdf` on the input
            PDF first to fix two-column reading order, then convert
            the segmented PDF.

    Returns:
        A :class:`ConversionResult` with paths and counts.
    """
    t0 = time.perf_counter()

    actual_source = source

    if auto_segment:
        from docberry.segmenter import segment_pdf as _segment_pdf

        seg_output = str(Path(source).with_name(Path(source).stem + "_segmented.pdf"))
        log.info("Auto-segmenting %s -> %s", source, seg_output)
        _segment_pdf(source, seg_output)
        actual_source = seg_output

    enrich_formulas_in_docling = equation_enrichment == "docling"

    converter = _build_converter(
        extract_assets, layout_model, pipeline, enrich_formulas_in_docling,
    )
    log.info("Converting: %s", actual_source)
    conv_res = converter.convert(actual_source)

    result = ConversionResult()

    if extract_assets:
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = Path(source).parent / f"{Path(source).stem}_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        result.output_dir = out_dir

        enrich_method = equation_enrichment if equation_enrichment != "docling" else "none"
        counts = _extract_assets(conv_res, out_dir, enrich_method)

        result.tables = counts["tables"]
        result.figures = counts["figures"]
        result.equations = counts["equations"]
        result.markdown_path = counts["md_path"]
    else:
        doc = conv_res.document
        if output_format == "json":
            content = repr(doc.export_to_dict())
            ext = ".json"
        else:
            content = doc.export_to_markdown()
            ext = ".md"

        if output_dir:
            out_path = Path(output_dir) / f"{Path(source).stem}{ext}"
        else:
            out_path = Path(source).with_suffix(ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        log.info("Saved output to: %s", out_path)

        if ext == ".md":
            result.markdown_path = out_path
        else:
            result.json_path = out_path

    result.elapsed_seconds = time.perf_counter() - t0
    log.info("Total time: %.2f s", result.elapsed_seconds)
    return result
