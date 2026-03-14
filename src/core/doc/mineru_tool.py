"""
MinerU document conversion tool.

Converts PDFs to structured JSON using MinerU (mineru[all]).
Supports three backends: pipeline (fastest), hybrid (recommended), vlm (most accurate).

Dependencies:
- mineru[all]: uv pip install -U "mineru[all]"

Environment variable overrides:
- MINERU_BACKEND: Parsing backend (pipeline / hybrid-auto-engine / vlm-auto-engine).
- MINERU_LANG: OCR language (en / ch, etc.).
- MINERU_PARSE_METHOD: Parsing method (auto / txt / ocr).
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from core.config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# Supported backends
BACKEND_PIPELINE = "pipeline"
BACKEND_HYBRID = "hybrid-auto-engine"
BACKEND_VLM = "vlm-auto-engine"
DEFAULT_BACKEND = BACKEND_HYBRID

_MINERU_IMPORT_ERROR: Optional[Exception] = None

try:
    from mineru.cli.common import read_fn

    _MINERU_AVAILABLE = True
except ImportError as exc:
    _MINERU_AVAILABLE = False
    _MINERU_IMPORT_ERROR = exc


def is_mineru_available() -> bool:
    """Return whether MinerU is available in the current environment."""
    return _MINERU_AVAILABLE


def get_mineru_install_hint() -> str:
    """Return an installation hint when MinerU is missing."""
    hint = 'MinerU is not installed. Please run: uv pip install -U "mineru[all]"'
    if _MINERU_IMPORT_ERROR is None:
        return hint
    return f"{hint}\nImport error: {_MINERU_IMPORT_ERROR}"


def _check_mineru() -> None:
    if not _MINERU_AVAILABLE:
        raise RuntimeError(get_mineru_install_hint())


def _analyze_pdf(
    pdf_bytes: bytes,
    backend: str = DEFAULT_BACKEND,
    lang: str = "en",
    parse_method: str = "auto",
    formula_enable: bool = True,
    table_enable: bool = True,
) -> dict:
    """Call a MinerU backend to parse PDF bytes and return middle_json."""
    from mineru.data.data_reader_writer import FileBasedDataWriter

    if backend == BACKEND_PIPELINE:
        from mineru.backend.pipeline.pipeline_analyze import (
            doc_analyze as pipeline_doc_analyze,
        )
        from mineru.backend.pipeline.model_json_to_middle_json import (
            result_to_middle_json,
        )

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                [pdf_bytes],
                [lang],
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            image_writer = FileBasedDataWriter(tmpdir)
            middle_json = result_to_middle_json(
                infer_results[0],
                all_image_lists[0],
                all_pdf_docs[0],
                image_writer,
                lang_list[0],
                ocr_enabled_list[0],
                formula_enable,
            )
        return middle_json

    elif backend.startswith("hybrid"):
        from mineru.backend.hybrid.hybrid_analyze import (
            doc_analyze as hybrid_doc_analyze,
        )
        from mineru.utils.engine_utils import get_vlm_engine

        engine = backend.removeprefix("hybrid-")
        if engine == "auto-engine":
            engine = get_vlm_engine(inference_engine="auto", is_async=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_writer = FileBasedDataWriter(tmpdir)
            middle_json, _infer_result, _ = hybrid_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                backend=engine,
                parse_method=f"hybrid_{parse_method}",
                language=lang,
                inline_formula_enable=formula_enable,
            )
        return middle_json

    elif backend.startswith("vlm"):
        from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
        from mineru.utils.engine_utils import get_vlm_engine

        engine = backend.removeprefix("vlm-")
        if engine == "auto-engine":
            engine = get_vlm_engine(inference_engine="auto", is_async=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_writer = FileBasedDataWriter(tmpdir)
            middle_json, _infer_result = vlm_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                backend=engine,
            )
        return middle_json

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _generate_content_list(middle_json: dict, backend: str) -> list:
    """Generate a flat content_list from middle_json, with page_idx per block."""
    from mineru.utils.enum_class import MakeMode

    pdf_info = middle_json["pdf_info"]
    if backend == BACKEND_PIPELINE:
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
    else:
        # Both hybrid and vlm use the vlm union_make
        from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make

    return union_make(pdf_info, MakeMode.CONTENT_LIST, "")


def to_json(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Convert a PDF to structured JSON using MinerU.

    Args:
        input_path: Path to the input PDF.
        output_path: Path for the output file (auto-generated if None).
        output_dir: Output directory.

    Returns:
        Path to the output file.
    """
    _check_mineru()

    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Read optional config from environment variables
    backend = os.environ.get("MINERU_BACKEND", DEFAULT_BACKEND)
    lang = os.environ.get("MINERU_LANG", "en")
    parse_method = os.environ.get("MINERU_PARSE_METHOD", "auto")

    logger.info(f"MinerU parsing: {input_file.name} (backend={backend})")

    pdf_bytes = input_file.read_bytes()

    middle_json = _analyze_pdf(
        pdf_bytes,
        backend=backend,
        lang=lang,
        parse_method=parse_method,
    )

    # Generate content_list
    content_list = _generate_content_list(middle_json, backend)
    logger.info(f"MinerU content_list: {len(content_list)} blocks")

    # Assemble output
    result = {
        "parser": "mineru",
        "backend": backend,
        "source_file": input_file.name,
        "content_list": content_list,
        "middle_json": middle_json,
    }

    # Determine output path
    if output_path is None:
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        result_dir = Path(output_dir)
        output_file = result_dir / f"{input_file.stem}_mineru.json"
    else:
        output_file = Path(output_path).expanduser().resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"MinerU output: {output_file}")
    return output_file


if __name__ == "__main__":
    input_pdf = "test.pdf"
    json_path = to_json(input_pdf)
    print(f"JSON saved to: {json_path}")
