"""Adapter: render PDF pages and write MDocAgent's expected sample layout.

MDocAgent's ``BaseDataset`` (mydatasets/base_dataset.py) reads:
    extract_path/<doc_name>_<page_idx>.png   -- 0-indexed PNG
    extract_path/<doc_name>_<page_idx>.txt   -- 0-indexed per-page text
    data_dir/samples.json                    -- list of sample dicts
    data_dir/sample-with-retrieval-results.json  -- samples + retrieval keys

The caller passes explicit ``data_dir`` and ``extract_dir``. Convention used by
``baseline.agentic_mdocagent``:

- ``extract_dir`` is SHARED across calls (`<upstream>/tmp/lsf/`), since the
  page renders are deterministic per ``(doc_name, page_idx)`` and benefit from
  caching across queries on the same doc.
- ``data_dir`` is PER-CALL (`<upstream>/data/run-<run_name>/`) and always
  contains exactly one sample, so the upstream ``predict_dataset`` loop runs
  on this (doc, question) pair only — no accumulation, no concurrency races.

Retrieval strategy: ColBERT/ColPali is bypassed; we pre-supply the first
N page indices (N <= ``_R_MAX_PAGES``) as both the text and image retrieved
sets. The "top-10" key name mirrors upstream's ``top_k=10`` retrieval config.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_UPSTREAM_DIR = Path(__file__).parent / "upstream" / "MDocAgent"
_RENDER_DPI = 144

_R_TEXT_KEY = "text-top-10-question"
_R_IMAGE_KEY = "image-top-10-question"
_R_MAX_PAGES = 10


def _doc_name_from_doc_id(doc_id: str) -> str:
    return Path(re.sub(r"\.pdf$", "", doc_id, flags=re.IGNORECASE)).name


def prepare_inputs(
    pdf_path: Path | str,
    doc_id: str,
    *,
    data_dir: Path,
    extract_dir: Path,
    query_idx: int = 0,
    query_text: str = "",
    max_pages: int | None = None,
) -> dict[str, Any]:
    """Materialise exactly one (query, doc) sample in MDocAgent's expected layout.

    ``data_dir`` is overwritten with a single-sample ``samples.json`` and a
    matching ``sample-with-retrieval-results.json``. ``extract_dir`` is
    populated with the doc's page renders (cache-aware: existing files are
    not re-rendered).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found for MDocAgent prep: {pdf_path}")

    data_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    doc_name = _doc_name_from_doc_id(doc_id)
    sample_id = f"{query_idx}_{doc_name}"

    n_pages = _render_pages(pdf_path, doc_name, extract_dir, max_pages=max_pages)
    _write_sample(
        data_dir=data_dir,
        sample_id=sample_id,
        doc_id=doc_id,
        query_text=query_text,
        n_pages=n_pages,
    )

    return {
        "sample_id": sample_id,
        "doc_name": doc_name,
        "n_pages": n_pages,
        "data_dir": data_dir,
        "extract_path": extract_dir,
    }


def _render_pages(
    pdf_path: Path,
    doc_name: str,
    extract_path: Path,
    max_pages: int | None,
) -> int:
    """Render PDF pages to PNG + extract per-page text via pymupdf.

    Returns the number of pages rendered. Existing files are not re-rendered
    so repeated calls for the same doc are cheap.
    """
    import pymupdf  # type: ignore[import]

    n_pages = 0
    with pymupdf.open(str(pdf_path)) as pdf:
        effective = min(len(pdf), max_pages) if max_pages is not None else len(pdf)
        for page_idx, page in enumerate(pdf):
            if page_idx >= effective:
                break
            img_file = extract_path / f"{doc_name}_{page_idx}.png"
            if not img_file.exists():
                pix = page.get_pixmap(dpi=_RENDER_DPI)
                pix.save(str(img_file))
            txt_file = extract_path / f"{doc_name}_{page_idx}.txt"
            if not txt_file.exists():
                txt_file.write_text(page.get_text("text"), encoding="utf-8")
            n_pages += 1
    return n_pages


def _load_json_list(path: Path) -> list[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def _save_json_list(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_sample_record(sample_id: str, doc_id: str, query_text: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "doc_id": doc_id,
        "question": query_text,
        "answer": "",
    }


def _build_retrieval_record(
    sample_id: str,
    doc_id: str,
    query_text: str,
    n_pages: int,
) -> dict[str, Any]:
    capped = list(range(min(n_pages, _R_MAX_PAGES)))
    record = _build_sample_record(sample_id, doc_id, query_text)
    record[_R_TEXT_KEY] = capped
    record[_R_IMAGE_KEY] = capped
    return record


def _write_sample(
    data_dir: Path,
    sample_id: str,
    doc_id: str,
    query_text: str,
    n_pages: int,
) -> None:
    """Overwrite ``data_dir`` with exactly one sample.

    ``data_dir`` is per-call, so we don't preserve any prior contents.
    """
    samples_path = data_dir / "samples.json"
    retrieval_path = data_dir / "sample-with-retrieval-results.json"
    _save_json_list(samples_path, [_build_sample_record(sample_id, doc_id, query_text)])
    _save_json_list(
        retrieval_path,
        [_build_retrieval_record(sample_id, doc_id, query_text, n_pages)],
    )
