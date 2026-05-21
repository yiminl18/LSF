"""LLM-OCR for the DeepRead baseline.

Each page is rendered to JPEG, sent to a vision-capable chat model, parsed into
section/paragraph records, and cached under `.cache/deepread_ocr/`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from baseline.deepread.index import ParagraphIndex
from baseline.deepread.llm import DEFAULT_MODEL, DEFAULT_PROVIDER, chat_vision

_ROOT = Path(__file__).resolve().parents[3]
_CACHE_DIR = _ROOT / ".cache" / "deepread_ocr"
_OCR_PROMPT_PATH = Path(__file__).with_name("deepread_ocr_page.txt")

DEFAULT_OCR_MODEL = DEFAULT_MODEL
DEFAULT_OCR_PROVIDER = DEFAULT_PROVIDER
_MAX_OCR_TOKENS = 2000
_RENDER_DPI = 100

_logger = logging.getLogger(__name__)


def _prompt_hash() -> str:
    return hashlib.sha256(_OCR_PROMPT_PATH.read_bytes()).hexdigest()[:10]


def _cache_path(
    doc_id: str,
    *,
    prompt_hash: str,
    model: str,
    max_pages: int | None,
) -> Path:
    safe_doc = re.sub(r"[^a-zA-Z0-9_.-]+", "_", doc_id)
    model_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
    pages_part = f"_maxpages{max_pages}" if max_pages is not None else ""
    return _CACHE_DIR / f"{safe_doc}__{prompt_hash}__{model_slug}{pages_part}.json"


def _load_ocr_prompt() -> str:
    return _OCR_PROMPT_PATH.read_text(encoding="utf-8")


def _render_page_jpeg(page: Any, dpi: int = _RENDER_DPI, quality: int = 85) -> bytes:
    import io

    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil().convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _parse_page_ocr(
    page_no: int,
    raw: str,
    global_section_counter: list[int],
) -> list[dict[str, Any]]:
    """Parse one page's structured Markdown into section dictionaries."""
    local_to_global: dict[str, int] = {}
    sections_out: list[dict[str, Any]] = []
    sections_by_gsid: dict[int, dict[str, Any]] = {}

    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    para_re = re.compile(r'<p\s+sid="([^"]+)"\s+pid="([^"]+)">(.*?)</p>', re.DOTALL)

    current_heading = ""
    current_level = 1

    events: list[tuple[int, str, Any]] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        hm = heading_re.match(line.rstrip("\r\n"))
        if hm:
            events.append((offset, "heading", hm))
        offset += len(line)
    for pm in para_re.finditer(raw):
        events.append((pm.start(), "paragraph", pm))
    events.sort(key=lambda e: e[0])

    for _, kind, match in events:
        if kind == "heading":
            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
            continue

        local_sid = match.group(1)
        text = match.group(3).strip()
        if not text:
            continue
        if local_sid not in local_to_global:
            global_section_counter[0] += 1
            gsid = global_section_counter[0]
            local_to_global[local_sid] = gsid
            section = {
                "section_id": gsid,
                "heading": current_heading or f"Section {gsid}",
                "level": current_level,
                "page_no": page_no,
                "paragraphs": [],
            }
            sections_out.append(section)
            sections_by_gsid[gsid] = section
        gsid = local_to_global[local_sid]
        sections_by_gsid[gsid]["paragraphs"].append({"text": text, "page_no": page_no})

    return sections_out


class LLMOCR:
    """LLM-based OCR that converts PDF pages to a ParagraphIndex."""

    def __init__(
        self,
        *,
        ocr_model: str = DEFAULT_OCR_MODEL,
        ocr_provider: str = DEFAULT_OCR_PROVIDER,
        max_pages: int | None = None,
    ) -> None:
        self._ocr_model = ocr_model
        self._ocr_provider = ocr_provider
        self._max_pages = max_pages
        self._last_cost_usd = 0.0
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_call_count = 0
        self._last_latency_seconds = 0.0
        self._last_cache_path: Path | None = None
        self._last_cache_hit = False

    @property
    def last_cost_usd(self) -> float:
        return self._last_cost_usd

    @property
    def last_input_tokens(self) -> int:
        return self._last_input_tokens

    @property
    def last_output_tokens(self) -> int:
        return self._last_output_tokens

    @property
    def last_call_count(self) -> int:
        return self._last_call_count

    @property
    def last_latency_seconds(self) -> float:
        return self._last_latency_seconds

    @property
    def last_cache_path(self) -> Path | None:
        return self._last_cache_path

    @property
    def last_cache_hit(self) -> bool:
        return self._last_cache_hit

    def parse_pdf(self, pdf_path: Path, doc_id: str) -> ParagraphIndex:
        cache = _cache_path(
            doc_id,
            prompt_hash=_prompt_hash(),
            model=self._ocr_model,
            max_pages=self._max_pages,
        )
        self._last_cache_path = cache
        self._last_cache_hit = False
        self._last_cost_usd = 0.0
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_call_count = 0
        self._last_latency_seconds = 0.0

        if cache.exists():
            self._last_cache_hit = True
            return ParagraphIndex.from_ocr_result(
                json.loads(cache.read_text(encoding="utf-8"))
            )

        index = self._parse_pdf_uncached(pdf_path)
        if not index.paragraphs:
            _logger.warning("OCR produced 0 paragraphs for %s; cache not written", pdf_path)
            return index

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(index.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return index

    def _parse_pdf_uncached(self, pdf_path: Path) -> ParagraphIndex:
        import pypdfium2 as pdfium  # type: ignore[import]

        prompt_template = _load_ocr_prompt()
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            n_pages = len(doc)
            effective_pages = (
                min(n_pages, self._max_pages)
                if self._max_pages is not None
                else n_pages
            )
            if self._max_pages is not None and n_pages > self._max_pages:
                _logger.info(
                    "max_pages=%d: capping OCR from %d to %d pages for %s",
                    self._max_pages,
                    n_pages,
                    effective_pages,
                    pdf_path.name,
                )

            all_sections: list[dict[str, Any]] = []
            global_section_counter = [0]
            for page_no in range(effective_pages):
                page = doc[page_no]
                jpeg_b64 = base64.b64encode(_render_page_jpeg(page)).decode("ascii")
                prompt_text = prompt_template.replace("{page_no}", str(page_no + 1))
                result = chat_vision(
                    prompt_text,
                    jpeg_b64,
                    provider=self._ocr_provider,
                    model=self._ocr_model,
                    max_tokens=_MAX_OCR_TOKENS,
                )
                self._last_cost_usd += result.cost_usd
                self._last_input_tokens += result.input_tokens
                self._last_output_tokens += result.output_tokens
                self._last_call_count += 1
                self._last_latency_seconds += result.latency_seconds
                all_sections.extend(
                    _parse_page_ocr(
                        page_no=page_no + 1,
                        raw=result.text,
                        global_section_counter=global_section_counter,
                    )
                )
        finally:
            doc.close()

        return ParagraphIndex.from_ocr_result({"sections": all_sections})
