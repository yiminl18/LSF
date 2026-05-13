"""LLM-OCR module for DeepRead.

Renders each PDF page to PNG via pypdfium2, then calls a vision-capable
LLM to produce hierarchical Markdown with structured paragraph tags.
The output is parsed into a ParagraphIndex and cached on disk.

Cache location: .cache/deepread_ocr/<doc_id>.json

Usage as a standalone prep step:
    PYTHONPATH=src python -m agent.baselines.deepread.ocr --query 0 --doc-id AMAZON_2015_10K
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from agent.baselines.deepread.index import ParagraphIndex
from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CACHE_DIR = _REPO_ROOT / ".cache" / "deepread_ocr"
_OCR_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "deepread_ocr_page.txt"
)

# Default vision model; override with --ocr-model.
# gpt-4o is required for vision; gpt-5.4-mini does not support image inputs.
_DEFAULT_OCR_MODEL = "gpt-4o"
_DEFAULT_OCR_PROVIDER = "azure"
_MAX_OCR_TOKENS = 2000
_RENDER_DPI = 100  # lower DPI to reduce token cost; sufficient for text extraction


def _cache_path(doc_id: str, max_pages: int | None = None) -> Path:
    if max_pages is not None:
        return _CACHE_DIR / f"{doc_id}_maxpages{max_pages}.json"
    return _CACHE_DIR / f"{doc_id}.json"


def _load_ocr_prompt() -> str:
    return _OCR_PROMPT_PATH.read_text(encoding="utf-8")


def _render_page_png(page: Any, dpi: int = _RENDER_DPI) -> bytes:
    """Render a pypdfium2 page to PNG bytes at the specified DPI."""
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def _parse_page_ocr(page_no: int, raw: str, global_section_counter: list[int]) -> list[dict[str, Any]]:
    """Parse one page's LLM-OCR output into a list of section/paragraph dicts.

    Expected LLM output per page:
      # Heading
      ## Sub-heading
      <p sid="1" pid="1">paragraph text</p>

    Returns a list of section dicts (same format as ParagraphIndex.from_ocr_result expects).
    sid values are page-local; we remap them to global monotonic IDs.
    """
    # Map page-local sid -> global section id
    local_to_global: dict[str, int] = {}
    sections_out: list[dict[str, Any]] = []
    current_section_local: str | None = None

    # Parse heading lines and paragraph tags
    lines = raw.splitlines()
    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    para_re = re.compile(r'<p\s+sid="([^"]+)"\s+pid="([^"]+)">(.*?)</p>', re.DOTALL)

    current_heading = ""
    current_level = 1
    para_order_in_section: dict[str, int] = {}

    # First pass: collect all paragraphs in order
    for line in lines:
        hm = heading_re.match(line)
        if hm:
            level = len(hm.group(1))
            heading = hm.group(2).strip()
            # This heading creates a new logical section; we don't have a sid yet
            current_heading = heading
            current_level = level
            continue
        # Inline paragraphs
        for pm in para_re.finditer(line):
            local_sid = pm.group(1)
            text = pm.group(3).strip()
            if not text:
                continue
            if local_sid not in local_to_global:
                global_section_counter[0] += 1
                local_to_global[local_sid] = global_section_counter[0]
                sections_out.append({
                    "section_id": global_section_counter[0],
                    "heading": current_heading or f"Section {global_section_counter[0]}",
                    "level": current_level,
                    "page_no": page_no,
                    "paragraphs": [],
                })
            gsid = local_to_global[local_sid]
            order = para_order_in_section.get(local_sid, 0)
            para_order_in_section[local_sid] = order + 1
            # Find the section in sections_out
            for sec in sections_out:
                if sec["section_id"] == gsid:
                    sec["paragraphs"].append({"text": text, "page_no": page_no})
                    break

    # Also scan for multi-line paragraph tags
    for pm in para_re.finditer(raw):
        local_sid = pm.group(1)
        text = pm.group(3).strip()
        if not text:
            continue
        if local_sid not in local_to_global:
            global_section_counter[0] += 1
            local_to_global[local_sid] = global_section_counter[0]
            sections_out.append({
                "section_id": global_section_counter[0],
                "heading": current_heading or f"Section {global_section_counter[0]}",
                "level": current_level,
                "page_no": page_no,
                "paragraphs": [],
            })
        gsid = local_to_global[local_sid]
        # Avoid adding duplicates already added by line scan
        for sec in sections_out:
            if sec["section_id"] == gsid:
                existing = [p["text"] for p in sec["paragraphs"]]
                if text not in existing:
                    sec["paragraphs"].append({"text": text, "page_no": page_no})
                break

    return sections_out


_ocr_logger = logging.getLogger(__name__)


class LLMOCR:
    """LLM-based OCR that converts PDF pages to a structured ParagraphIndex."""

    def __init__(
        self,
        cached_caller: CachedLLMCaller,
        ocr_model: str = _DEFAULT_OCR_MODEL,
        ocr_provider: str = _DEFAULT_OCR_PROVIDER,
        max_pages: int | None = None,
    ) -> None:
        self._caller = cached_caller
        self._ocr_model = ocr_model
        self._ocr_provider = ocr_provider
        self._max_pages = max_pages

    def parse_pdf(self, pdf_path: Path, doc_id: str) -> ParagraphIndex:
        """Parse pages of pdf_path; uses disk cache when available.

        Cache key is doc_id-specific; max_pages is included in the filename to
        avoid poisoning the full-run cache with a capped partial result.
        """
        cache = _cache_path(doc_id, self._max_pages)
        if cache.exists():
            with cache.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return ParagraphIndex.from_ocr_result(data)

        result = self._parse_pdf_uncached(pdf_path)

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = result.to_dict()
        with cache.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return result

    def _parse_pdf_uncached(self, pdf_path: Path) -> ParagraphIndex:
        import pypdfium2 as pdfium  # type: ignore[import]

        prompt_template = _load_ocr_prompt()
        doc = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(doc)
        effective_pages = min(n_pages, self._max_pages) if self._max_pages is not None else n_pages
        if self._max_pages is not None and n_pages > self._max_pages:
            _ocr_logger.info(
                "max_pages=%d: capping OCR from %d to %d pages for %s",
                self._max_pages, n_pages, effective_pages, pdf_path.name,
            )

        global_section_counter = [0]  # mutable reference for page-local -> global mapping
        all_sections: list[dict[str, Any]] = []

        for page_no in range(effective_pages):
            page = doc[page_no]
            png_bytes = _render_page_png(page)
            b64 = base64.b64encode(png_bytes).decode("ascii")
            prompt = prompt_template.replace("{page_no}", str(page_no + 1)).replace(
                "{image_b64}", b64
            )
            result = self._caller.call(
                prompt,
                llm_provider=self._ocr_provider,
                max_tokens=_MAX_OCR_TOKENS,
                model=self._ocr_model,
            )
            page_sections = _parse_page_ocr(
                page_no=page_no + 1,
                raw=result.response,
                global_section_counter=global_section_counter,
            )
            all_sections.extend(page_sections)

        doc.close()
        return ParagraphIndex.from_ocr_result({"sections": all_sections})


def main(argv: list[str] | None = None) -> None:
    """Standalone prep step: pre-OCR PDFs for DeepRead."""
    import yaml

    parser = argparse.ArgumentParser(description="Pre-OCR PDFs for DeepRead baseline")
    parser.add_argument("--config", type=Path, default=Path("src/agent/config_pdfs_10doc.yaml"))
    parser.add_argument("--query", type=int, required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--ocr-model", default=_DEFAULT_OCR_MODEL)
    parser.add_argument("--ocr-provider", default=_DEFAULT_OCR_PROVIDER)
    parser.add_argument("--max-pages", type=int, default=None, help="Cap OCR to this many pages")
    args = parser.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config.get("dataset_root", "datasets/pdfs/latest"))
    pdf_path = dataset_root / "raw" / f"{args.doc_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cached_caller = CachedLLMCaller(DEFAULT_CACHE_DB_PATH)
    ocr = LLMOCR(
        cached_caller,
        ocr_model=args.ocr_model,
        ocr_provider=args.ocr_provider,
        max_pages=args.max_pages,
    )
    idx = ocr.parse_pdf(pdf_path, doc_id=args.doc_id)
    print(f"OCR complete: {len(idx.paragraphs)} paragraphs, {len(idx.sections)} sections")
    cache = _cache_path(args.doc_id)
    print(f"Cached at: {cache}")


if __name__ == "__main__":
    main()
