"""LLM-OCR module for DeepRead.

Renders each PDF page to JPEG via pypdfium2, then calls a vision-capable
LLM to produce hierarchical Markdown with structured paragraph tags. The
output is parsed into a ParagraphIndex and cached on disk.

Cache location: .cache/deepread_ocr/<doc_id>__<prompt_hash>__<model>.json
The prompt hash and model are baked into the filename so that editing the
OCR prompt or switching the OCR model invalidates the cache automatically.

Usage as a standalone prep step:
    PYTHONPATH=src python -m agent.baselines.deepread.ocr --query 0 --doc-id AMAZON_2015_10K
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from agent.baselines.defaults import (
    DEFAULT_OCR_MODEL as _DEFAULT_OCR_MODEL,
    DEFAULT_OCR_PROVIDER as _DEFAULT_OCR_PROVIDER,
)
from agent.baselines.deepread.index import ParagraphIndex
from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH
from core.llm.cost import compute_cost

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CACHE_DIR = _REPO_ROOT / ".cache" / "deepread_ocr"
_OCR_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "baselines" / "deepread_ocr_page.txt"
)

# Re-export so extractor.py and the standalone CLI keep using one symbol.
DEFAULT_OCR_MODEL = _DEFAULT_OCR_MODEL
DEFAULT_OCR_PROVIDER = _DEFAULT_OCR_PROVIDER
_MAX_OCR_TOKENS = 2000
_RENDER_DPI = 100  # lower DPI to reduce token cost; sufficient for text extraction


def _prompt_hash() -> str:
    text = _OCR_PROMPT_PATH.read_bytes()
    return hashlib.sha256(text).hexdigest()[:10]


def _cache_path(
    doc_id: str,
    *,
    prompt_hash: str,
    model: str,
    max_pages: int | None,
) -> Path:
    model_slug = model.replace("/", "_")
    pages_part = f"_maxpages{max_pages}" if max_pages is not None else ""
    return _CACHE_DIR / f"{doc_id}__{prompt_hash}__{model_slug}{pages_part}.json"


def _load_ocr_prompt() -> str:
    return _OCR_PROMPT_PATH.read_text(encoding="utf-8")


def _render_page_jpeg(page: Any, dpi: int = _RENDER_DPI, quality: int = 85) -> bytes:
    """Render a pypdfium2 page to JPEG bytes."""
    import io
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil().convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _ocr_page_call(
    prompt_text: str,
    jpeg_b64: str,
    provider: str,
    model: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Make a vision API call for one page. Returns (response_text, in_tokens, out_tokens)."""
    import os
    from openai import OpenAI, AzureOpenAI

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"},
                },
            ],
        }
    ]

    if provider == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=0,
        )
    else:  # azure
        client = AzureOpenAI(  # type: ignore[assignment]
            azure_endpoint=os.environ["AZURE_54MINI_API_BASE"],
            api_key=os.environ["AZURE_54MINI_API_KEY"],
            api_version=os.environ.get("AZURE_54MINI_API_VERSION", "2024-12-01-preview"),
        )
        resp = client.chat.completions.create(
            model=os.environ.get("AZURE_54MINI_DEPLOYMENT", model),
            messages=messages,  # type: ignore[arg-type]
            max_completion_tokens=max_tokens,
            temperature=0,
        )

    text = resp.choices[0].message.content or ""
    usage = resp.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    return text, in_tok, out_tok


def _parse_page_ocr(page_no: int, raw: str, global_section_counter: list[int]) -> list[dict[str, Any]]:
    """Parse one page's LLM-OCR output into a list of section/paragraph dicts.

    Expected LLM output per page:
      # Heading
      ## Sub-heading
      <p sid="1" pid="1">paragraph text</p>

    sid values are page-local; we remap them to global monotonic IDs.
    """
    import re

    local_to_global: dict[str, int] = {}
    sections_out: list[dict[str, Any]] = []
    sections_by_gsid: dict[int, dict[str, Any]] = {}

    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    para_re = re.compile(r'<p\s+sid="([^"]+)"\s+pid="([^"]+)">(.*?)</p>', re.DOTALL)

    current_heading = ""
    current_level = 1

    # Walk the raw output line-by-line so heading state stays correct,
    # but match paragraph tags greedily across lines so multi-line <p>...</p>
    # bodies are captured. One pass — no need to scan twice.
    cursor = 0
    while cursor < len(raw):
        nl = raw.find("\n", cursor)
        line_end = nl if nl != -1 else len(raw)
        line = raw[cursor:line_end]

        hm = heading_re.match(line)
        if hm:
            current_level = len(hm.group(1))
            current_heading = hm.group(2).strip()
            cursor = line_end + 1
            continue

        m = para_re.search(raw, cursor)
        if m is None:
            break
        local_sid = m.group(1)
        text = m.group(3).strip()
        if not text:
            cursor = m.end()
            continue
        if local_sid not in local_to_global:
            global_section_counter[0] += 1
            gsid = global_section_counter[0]
            local_to_global[local_sid] = gsid
            section_dict = {
                "section_id": gsid,
                "heading": current_heading or f"Section {gsid}",
                "level": current_level,
                "page_no": page_no,
                "paragraphs": [],
            }
            sections_out.append(section_dict)
            sections_by_gsid[gsid] = section_dict
        gsid = local_to_global[local_sid]
        sections_by_gsid[gsid]["paragraphs"].append({"text": text, "page_no": page_no})
        cursor = m.end()

    return sections_out


_ocr_logger = logging.getLogger(__name__)


class LLMOCR:
    """LLM-based OCR that converts PDF pages to a structured ParagraphIndex."""

    def __init__(
        self,
        cached_caller: CachedLLMCaller,
        ocr_model: str = DEFAULT_OCR_MODEL,
        ocr_provider: str = DEFAULT_OCR_PROVIDER,
        max_pages: int | None = None,
    ) -> None:
        # cached_caller is accepted for interface symmetry with other baselines,
        # but vision multimodal calls don't pass through SQLite (multipart
        # payloads don't share cache keys with text prompts). OCR has its own
        # per-document JSON cache instead.
        self._caller = cached_caller
        self._ocr_model = ocr_model
        self._ocr_provider = ocr_provider
        self._max_pages = max_pages
        self._last_cost_usd: float = 0.0
        self._last_call_count: int = 0

    @property
    def last_cost_usd(self) -> float:
        """Cost (USD) of the most recent parse_pdf() call. 0 when fully cached."""
        return self._last_cost_usd

    @property
    def last_call_count(self) -> int:
        """Number of vision calls made by the most recent parse_pdf(). 0 when cached."""
        return self._last_call_count

    def parse_pdf(self, pdf_path: Path, doc_id: str) -> ParagraphIndex:
        """Parse pages of pdf_path; uses disk cache when available.

        Cache key includes the prompt hash and model so the cache invalidates
        automatically when either changes. Empty results are not cached, so
        a transient LLM hiccup doesn't poison future runs.
        """
        cache = _cache_path(
            doc_id,
            prompt_hash=_prompt_hash(),
            model=self._ocr_model,
            max_pages=self._max_pages,
        )
        self._last_cost_usd = 0.0
        self._last_call_count = 0
        if cache.exists():
            with cache.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return ParagraphIndex.from_ocr_result(data)

        result, cost_usd, call_count = self._parse_pdf_uncached(pdf_path)
        self._last_cost_usd = cost_usd
        self._last_call_count = call_count

        if not result.paragraphs:
            _ocr_logger.warning(
                "OCR produced 0 paragraphs for %s; not writing cache to avoid poisoning",
                pdf_path.name,
            )
            return result

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = result.to_dict()
        with cache.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return result

    def _parse_pdf_uncached(self, pdf_path: Path) -> tuple[ParagraphIndex, float, int]:
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

        global_section_counter = [0]
        all_sections: list[dict[str, Any]] = []
        total_cost = 0.0
        call_count = 0

        for page_no in range(effective_pages):
            page = doc[page_no]
            jpeg_bytes = _render_page_jpeg(page)
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            prompt_text = prompt_template.replace("{page_no}", str(page_no + 1))
            response_text, in_tok, out_tok = _ocr_page_call(
                prompt_text=prompt_text,
                jpeg_b64=b64,
                provider=self._ocr_provider,
                model=self._ocr_model,
                max_tokens=_MAX_OCR_TOKENS,
            )
            total_cost += compute_cost(in_tok, out_tok, self._ocr_provider, model=self._ocr_model)
            call_count += 1
            page_sections = _parse_page_ocr(
                page_no=page_no + 1,
                raw=response_text,
                global_section_counter=global_section_counter,
            )
            all_sections.extend(page_sections)

        doc.close()
        return ParagraphIndex.from_ocr_result({"sections": all_sections}), total_cost, call_count


def main(argv: list[str] | None = None) -> None:
    """Standalone prep step: pre-OCR PDFs for DeepRead."""
    import yaml

    parser = argparse.ArgumentParser(description="Pre-OCR PDFs for DeepRead baseline")
    parser.add_argument("--config", type=Path, default=Path("src/agent/config_pdfs_10doc.yaml"))
    parser.add_argument("--query", type=int, required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--ocr-model", default=DEFAULT_OCR_MODEL)
    parser.add_argument("--ocr-provider", default=DEFAULT_OCR_PROVIDER)
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
    print(f"Cost: ${ocr.last_cost_usd:.4f}")
    cache = _cache_path(
        args.doc_id,
        prompt_hash=_prompt_hash(),
        model=args.ocr_model,
        max_pages=args.max_pages,
    )
    print(f"Cached at: {cache}")


if __name__ == "__main__":
    main()
