"""
Lazy-loaded equation enrichment backends.

Each backend is loaded only when first requested so that optional
dependencies (pix2tex, torch, transformers) are not required at
import time.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from PIL import Image as PILImage

log = logging.getLogger(__name__)


def init_pix2tex() -> Any:
    """Load the pix2tex LatexOCR model.

    Raises ``SystemExit`` with a helpful message if pix2tex is not installed.
    """
    try:
        from pix2tex.cli import LatexOCR
        model = LatexOCR()
        log.info("pix2tex LatexOCR model loaded for lite formula enrichment")
        return model
    except ImportError:
        log.error(
            "pix2tex is not installed. Install it with: pip install 'docberry[pix2tex]'"
        )
        raise SystemExit(1)


def init_qwen_vlm() -> Tuple[Any, Any]:
    """Load the Qwen3.5-0.8B VLM for equation LaTeX extraction.

    Returns ``(model, processor)``.

    Raises ``SystemExit`` with a helpful message if torch/transformers
    are not installed.
    """
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = "Qwen/Qwen3.5-0.8B"
        log.info("Loading Qwen3.5-0.8B model (first run downloads ~1.6 GB)...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        log.info("Qwen3.5-0.8B model loaded for equation enrichment")
        return model, processor
    except ImportError as exc:
        log.error(
            "transformers or torch not available for Qwen VLM: %s. "
            "Install with: pip install 'docberry[qwen]'",
            exc,
        )
        raise SystemExit(1)


def qwen_image_to_latex(
    img: PILImage.Image, model: Any, processor: Any,
) -> str:
    """Run Qwen3.5-0.8B inference on a single equation image."""
    import torch

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": (
            "Convert this mathematical equation image to LaTeX. "
            "Output only the raw LaTeX formula, with no explanation, "
            "no surrounding text, and no $$ delimiters."
        )},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=[img], return_tensors="pt", padding=True,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    output = processor.decode(
        generated_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return output.strip()


def download_pix2tex_model() -> None:
    """Pre-download pix2tex model weights."""
    try:
        from pix2tex.cli import LatexOCR
        log.info("Downloading pix2tex model weights...")
        LatexOCR()
        log.info("pix2tex model weights downloaded successfully.")
    except ImportError:
        log.error("pix2tex is not installed. Install with: pip install 'docberry[pix2tex]'")
        raise SystemExit(1)


def download_qwen_model() -> None:
    """Pre-download Qwen3.5-0.8B model weights."""
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model_id = "Qwen/Qwen3.5-0.8B"
        log.info("Downloading Qwen3.5-0.8B model weights (~1.6 GB)...")
        AutoModelForImageTextToText.from_pretrained(model_id)
        AutoProcessor.from_pretrained(model_id)
        log.info("Qwen3.5-0.8B model weights downloaded successfully.")
    except ImportError:
        log.error("transformers is not installed. Install with: pip install 'docberry[qwen]'")
        raise SystemExit(1)


def download_docling_models() -> None:
    """Pre-download Docling layout models used by the standard pipeline."""
    log.info("Downloading Docling layout models (triggered by a dummy import)...")
    try:
        from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON  # noqa: F401
        log.info(
            "Docling models are downloaded on first conversion. "
            "Run a quick test conversion to trigger the download."
        )
    except ImportError:
        log.error("docling is not installed. Install with: pip install docberry")
        raise SystemExit(1)
