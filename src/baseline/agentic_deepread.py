"""Baseline: DeepRead OCR + locate/read QA.

This is a lightweight port of chiyu-dev's DeepRead baseline to yiming-dev's
`src/baseline/run_eval.py` contract. Public surface:

    run_qa(doc_path, question, *, model, timeout, log_dir, log_stem, **kwargs)

DeepRead reads PDFs directly, builds a cached LLM-OCR paragraph index, then
runs a locate/read loop over Retrieve and ReadSection tools.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from baseline.deepread.extractor import DeepReadExtractor
from baseline.deepread.llm import DEFAULT_MODEL, DEFAULT_PROVIDER, resolve_model
from baseline.deepread.ocr import DEFAULT_OCR_MODEL, DEFAULT_OCR_PROVIDER

SUPPORTS_PDF_INPUT = True

_PDF_DIR_DEFAULTS: dict[str, Path] = {
    "nopv": _ROOT / "data/nopv/raw",
    "financebench": _ROOT / "data/financebench/raw",
}


def _doc_name_from_path(path: Path) -> str:
    return re.sub(r"_reconstructed$", "", path.stem)


def _resolve_pdf_path(doc_path: Path, pdf_dir: Path | None = None) -> Path:
    if doc_path.suffix.lower() == ".pdf":
        return doc_path

    doc_stem = _doc_name_from_path(doc_path)
    candidates: list[Path] = []
    if pdf_dir is not None:
        candidates.append(pdf_dir / f"{doc_stem}.pdf")
    candidates.extend(d / f"{doc_stem}.pdf" for d in _PDF_DIR_DEFAULTS.values())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No PDF found for doc_stem={doc_stem!r}. Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def _extract_text_from_json(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    texts = payload.get("texts")
    if not isinstance(texts, list):
        return ""
    parts: list[str] = []
    for item in texts:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        elif isinstance(item, str) and item.strip():
            parts.append(item.strip())
    return "\n\n".join(parts)


def _extract_text_from_pdf(path: Path, max_pages: int | None) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
        pages = list(reader.pages)
        if max_pages is not None:
            pages = pages[:max_pages]
        return "\n\n".join((page.extract_text() or "").strip() for page in pages)
    except Exception:
        return ""


def _safe_repo_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def run_qa(
    doc_path: str | Path,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
    log_dir: str | Path | None = None,
    log_stem: str | None = None,
    pdf_dir: str | Path | None = None,
    max_pages: int | None = None,
    ocr_model: str | None = None,
    ocr_provider: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    **_: Any,
) -> dict:
    """Run DeepRead on one (doc, question) pair and return run_eval telemetry."""
    del timeout  # Kept for the shared run_eval signature; no overall deadline is enforced here.

    resolved_model = resolve_model(model)
    source_path = Path(doc_path)
    try:
        pdf_path = _resolve_pdf_path(source_path, Path(pdf_dir) if pdf_dir else None)
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "answer": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "total_cost_usd": None,
            "model": resolved_model,
            "error_message": str(exc),
        }

    if not pdf_path.exists():
        return {
            "status": "error",
            "answer": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "total_cost_usd": None,
            "model": resolved_model,
            "error_message": f"PDF not found: {pdf_path}",
        }

    fallback_text = (
        _extract_text_from_json(source_path)
        if source_path.suffix.lower() == ".json"
        else _extract_text_from_pdf(pdf_path, max_pages=max_pages)
    )

    extractor = DeepReadExtractor(
        ocr_model=ocr_model or DEFAULT_OCR_MODEL,
        ocr_provider=ocr_provider or DEFAULT_OCR_PROVIDER,
        max_pages=max_pages,
    )

    try:
        result = extractor.extract(
            pdf_path=pdf_path,
            question=question,
            doc_id=pdf_path.stem,
            model=resolved_model,
            provider=provider,
            fallback_text=fallback_text,
        )
    except Exception as exc:
        return {
            "status": "error",
            "answer": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "total_cost_usd": None,
            "model": resolved_model,
            "error_message": str(exc),
        }

    log_path: Path | None = None
    if log_dir is not None and log_stem:
        log_base = Path(log_dir)
        log_base.mkdir(parents=True, exist_ok=True)
        log_path = log_base / f"{log_stem}.deepread.json"
        log_path.write_text(
            json.dumps(result.trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "status": "ok" if result.answer is not None else "no_answer",
        "answer": result.answer,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_seconds": result.latency_seconds,
        "total_cost_usd": result.total_cost_usd,
        "model": result.model,
        "gen_calls": result.gen_calls,
        "deepread_provider": provider,
        "deepread_ocr_provider": ocr_provider or DEFAULT_OCR_PROVIDER,
        "deepread_ocr_model": ocr_model or DEFAULT_OCR_MODEL,
        "deepread_max_pages": max_pages,
        "deepread_trace": result.trace,
        "deepread_log_path": _safe_repo_path(log_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepRead baseline - single pair")
    ap.add_argument("--doc", required=True, help="Path to a PDF or reconstructed JSON")
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Reader model alias or ID")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, help="Reader provider")
    ap.add_argument("--ocr-model", default=None, help="OCR vision model alias or ID")
    ap.add_argument("--ocr-provider", default=None, help="OCR provider")
    ap.add_argument("--pdf-dir", default=None)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    result = run_qa(
        args.doc,
        args.question,
        model=args.model,
        timeout=args.timeout,
        provider=args.provider,
        ocr_model=args.ocr_model,
        ocr_provider=args.ocr_provider,
        pdf_dir=args.pdf_dir,
        max_pages=args.max_pages,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
