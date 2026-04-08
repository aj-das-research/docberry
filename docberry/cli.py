"""
DocBerry command-line interface.

Entry point registered as ``docberry`` in pyproject.toml.

Subcommands::

    docberry convert paper.pdf -o output/ --extract-assets
    docberry segment paper.pdf -o paper_segmented.pdf
    docberry download-models --all
"""

from __future__ import annotations

import argparse
import logging
import sys


def _add_convert_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "convert",
        help="Convert a document to Markdown/JSON with optional asset extraction.",
    )
    p.add_argument("source", help="Path to a local file or a URL to convert.")
    p.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory. Defaults to <source_stem>_output/ next to the source.",
    )
    p.add_argument(
        "--format", choices=["markdown", "json"], default="markdown",
        dest="output_format",
        help="Output format (default: markdown).",
    )
    p.add_argument(
        "--extract-assets", action="store_true",
        help="Extract tables, figures, and equations as separate files.",
    )
    p.add_argument(
        "--layout-model", default="heron",
        choices=["heron", "egret-medium", "egret-large", "egret-xlarge"],
        help="Layout detection model (default: heron).",
    )
    p.add_argument(
        "--pipeline", choices=["standard", "vlm"], default="standard",
        help="Conversion pipeline: 'standard' or 'vlm'.",
    )
    p.add_argument(
        "--auto-segment", action="store_true",
        help="Segment the PDF for reading order before conversion.",
    )

    eq_group = p.add_mutually_exclusive_group()
    eq_group.add_argument(
        "--equation-enrichment", default="none",
        choices=["none", "pix2tex", "qwen", "docling"],
        help="Equation LaTeX extraction method (default: none).",
    )


def _add_segment_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "segment",
        help="Segment a two-column PDF into reading-order pages.",
    )
    p.add_argument("input", help="Input PDF path.")
    p.add_argument(
        "--output", "-o", default="segmented_output.pdf",
        help="Output segmented PDF path.",
    )
    p.add_argument("--pages", default=None, help="Page spec, 0-based (e.g. '0,2-4').")
    p.add_argument("--debug-dir", default=None, help="Directory for debug overlay images.")

    g = p.add_argument_group("layout tuning")
    g.add_argument("--line-merge-gap", type=float, default=3.0)
    g.add_argument("--band-merge-gap", type=float, default=14.0)
    g.add_argument("--block-padding", type=float, default=2.0)
    g.add_argument("--segment-padding", type=float, default=3.0)
    g.add_argument("--min-text-height", type=float, default=5.0)
    g.add_argument("--line-split-gap-x", type=float, default=14.0)
    g.add_argument("--num-bins", type=int, default=120)
    g.add_argument("--single-coverage-threshold", type=float, default=0.68)
    g.add_argument("--min-side-coverage", type=float, default=0.55)
    g.add_argument("--min-center-gap-ratio", type=float, default=0.08)
    g.add_argument("--min-band-height-ratio", type=float, default=0.018)
    g.add_argument("--caption-merge-gap", type=float, default=35.0)
    g.add_argument("--hrule-coverage", type=float, default=0.60)


def _add_download_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "download-models",
        help="Pre-download model weights used by DocBerry.",
    )
    p.add_argument(
        "--all", action="store_true", dest="download_all",
        help="Download all model weights (Docling + pix2tex + Qwen).",
    )
    p.add_argument(
        "--pix2tex", action="store_true",
        help="Download pix2tex model weights.",
    )
    p.add_argument(
        "--qwen", action="store_true",
        help="Download Qwen3.5-0.8B model weights.",
    )


def _run_convert(args: argparse.Namespace) -> None:
    from docberry.converter import convert_document

    result = convert_document(
        source=args.source,
        output_dir=args.output_dir,
        output_format=args.output_format,
        extract_assets=args.extract_assets,
        layout_model=args.layout_model,
        pipeline=args.pipeline,
        equation_enrichment=args.equation_enrichment,
        auto_segment=args.auto_segment,
    )

    print(f"\nConversion complete ({result.elapsed_seconds:.2f}s)")
    if result.markdown_path:
        print(f"  Markdown:  {result.markdown_path}")
    if result.json_path:
        print(f"  JSON:      {result.json_path}")
    if result.output_dir:
        print(f"  Output:    {result.output_dir}/")
        print(f"  Tables:    {result.tables}")
        print(f"  Figures:   {result.figures}")
        print(f"  Equations: {result.equations}")


def _run_segment(args: argparse.Namespace) -> None:
    from docberry.segmenter import segment_pdf, LayoutConfig

    config = LayoutConfig(
        line_merge_gap=args.line_merge_gap,
        band_merge_gap=args.band_merge_gap,
        block_padding=args.block_padding,
        segment_padding=args.segment_padding,
        min_text_height=args.min_text_height,
        line_split_gap_x=args.line_split_gap_x,
        num_bins=args.num_bins,
        single_coverage_threshold=args.single_coverage_threshold,
        min_side_coverage=args.min_side_coverage,
        min_center_gap_ratio=args.min_center_gap_ratio,
        min_band_height_ratio=args.min_band_height_ratio,
        caption_merge_gap=args.caption_merge_gap,
        hrule_coverage=args.hrule_coverage,
    )
    segments = segment_pdf(
        input_pdf=args.input,
        output_pdf=args.output,
        page_spec=args.pages,
        debug_dir=args.debug_dir,
        config=config,
    )
    print(f"\nWrote {len(segments)} segments to {args.output}\n")
    for seg in segments:
        print(f"  src_page_{seg.page_index:04d}  seg_{seg.order:04d}  {seg.label}")


def _run_download(args: argparse.Namespace) -> None:
    from docberry._enrichment import (
        download_docling_models,
        download_pix2tex_model,
        download_qwen_model,
    )

    if args.download_all:
        download_docling_models()
        download_pix2tex_model()
        download_qwen_model()
        return

    download_docling_models()

    if args.pix2tex:
        download_pix2tex_model()
    if args.qwen:
        download_qwen_model()

    print("\nModel download complete.")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="docberry",
        description="DocBerry — Extract structured Markdown, tables, figures, and equations from scientific PDFs.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")
    _add_convert_parser(subparsers)
    _add_segment_parser(subparsers)
    _add_download_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "convert":
        _run_convert(args)
    elif args.command == "segment":
        _run_segment(args)
    elif args.command == "download-models":
        _run_download(args)
    else:
        parser.print_help()
        sys.exit(1)


def _get_version() -> str:
    try:
        from docberry import __version__
        return __version__
    except ImportError:
        return "0.1.1"


if __name__ == "__main__":
    main()
