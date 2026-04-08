"""
DocBerry — Extract structured Markdown, tables, figures, and equations
from scientific PDFs with proper reading order.

Quick start::

    from docberry import segment_pdf, convert_document

    # Optional: segment a two-column PDF into reading order
    segment_pdf("paper.pdf", "paper_segmented.pdf")

    # Convert to Markdown with full asset extraction
    result = convert_document(
        "paper_segmented.pdf",
        output_dir="output/",
        extract_assets=True,
        equation_enrichment="pix2tex",
    )
    print(result.markdown_path, result.tables, result.figures, result.equations)
"""

__version__ = "0.1.2"

from docberry.segmenter import segment_pdf, LayoutConfig
from docberry.converter import convert_document, ConversionResult

__all__ = [
    "__version__",
    "segment_pdf",
    "LayoutConfig",
    "convert_document",
    "ConversionResult",
]
