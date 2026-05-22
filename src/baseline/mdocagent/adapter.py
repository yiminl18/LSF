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

Retrieval strategy: ColBERT/ColPali is bypassed. When ``rank_bm25`` is
installed and a non-empty ``query_text`` is given, we BM25-rank all per-page
TXTs and supply the top-K page indices to the text retrieval key (CPU-only,
no model download). When BM25 is unavailable we fall back to the first-N
pages. The image retrieval key is intentionally left empty: the LSF
integration routes ``image_agent`` to ``NoOpModel`` (see
``baseline.mdocagent.noop_model``), so populating image-top-10 would just be
a misleading duplicate of the text rank. The "top-10" key name mirrors
upstream's ``top_k=10`` retrieval config.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_UPSTREAM_DIR = Path(__file__).parent / "upstream" / "MDocAgent"
_RENDER_DPI = 144

# Single source of truth for the BM25 top-K page count. The key names mirror
# upstream's retrieval template (``text-top-${retrieval.top_k}-${...}``); we
# build them with the same K via f-string so changing _R_MAX_PAGES alone keeps
# both sides aligned. The matching upstream side is the
# ``retrieval.top_k=_R_MAX_PAGES`` Hydra override in
# ``agentic_mdocagent._agent_model_overrides`` — without that, upstream would
# interpolate against its default top_k=10 and our keys would drift.
_R_MAX_PAGES = 5
_R_TEXT_KEY = f"text-top-{_R_MAX_PAGES}-question"
_R_IMAGE_KEY = f"image-top-{_R_MAX_PAGES}-question"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer used for both the per-page corpus and
    the query during BM25 ranking. Numbers survive — Treasury Bulletin tables
    contain a lot of meaningful numeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _bm25_top_k(
    extract_dir: Path,
    doc_name: str,
    n_pages: int,
    query_text: str,
    k: int,
) -> list[int] | None:
    """Return the page indices of the top-K BM25 matches for ``query_text``.

    Returns ``None`` if ``rank_bm25`` isn't installed (the caller should fall
    back to the first-K pages). Pages whose text file is missing or empty
    contribute an empty token list — they can still be selected if no better
    page exists, but rank lower than any page with overlapping terms.
    """
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import]
    except ImportError:
        return None

    page_tokens: list[list[str]] = []
    for page_idx in range(n_pages):
        txt_file = extract_dir / f"{doc_name}_{page_idx}.txt"
        try:
            page_tokens.append(_tokenize(txt_file.read_text(encoding="utf-8")))
        except OSError:
            page_tokens.append([])

    # BM25Okapi requires every doc to be non-empty for the IDF computation;
    # substitute a single placeholder token for empty pages so the indexer
    # builds, but those pages get a uniform near-zero score.
    safe_pages = [tokens or ["_empty_"] for tokens in page_tokens]
    bm25 = BM25Okapi(safe_pages)
    scores = bm25.get_scores(_tokenize(query_text))
    ranked = sorted(range(n_pages), key=lambda i: scores[i], reverse=True)
    return ranked[:k]


_UNSAFE_DOC_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _doc_name_from_doc_id(doc_id: str) -> str:
    """Strip ``.pdf``, take the basename, and replace any chars that aren't
    safe in filenames or Hydra interpolation. Upstream's BaseDataset uses
    ``doc_id`` raw to build PNG/TXT paths, so this prefix must survive shells,
    POSIX filesystems, and OmegaConf ``${...}`` evaluation."""
    raw = Path(re.sub(r"\.pdf$", "", doc_id, flags=re.IGNORECASE)).name
    return _UNSAFE_DOC_NAME_RE.sub("_", raw)


def prepare_inputs(
    pdf_path: Path | str,
    doc_id: str,
    *,
    data_dir: Path,
    extract_dir: Path,
    query_idx: int = 0,
    query_text: str = "",
    max_pages: int | None = None,
    top_k: int = _R_MAX_PAGES,
) -> dict[str, Any]:
    """Materialise exactly one (query, doc) sample in MDocAgent's expected layout.

    ``data_dir`` is overwritten with a single-sample ``samples.json`` and a
    matching ``sample-with-retrieval-results.json``. ``extract_dir`` is
    populated with the doc's page renders (cache-aware: existing files are
    not re-rendered). When ``query_text`` is non-empty and ``rank_bm25`` is
    importable, the retrieval keys are populated with the BM25 top-K pages
    instead of the first K; otherwise it falls back to the first K.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found for MDocAgent prep: {pdf_path}")

    data_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    doc_name = _doc_name_from_doc_id(doc_id)
    # The samples.json doc_id is what upstream's BaseDataset.EXTRACT_DOCUMENT_ID
    # parses back into the PNG/TXT filename prefix; it must match ``doc_name``
    # exactly, so use the sanitised form here too.
    sanitized_doc_id = f"{doc_name}.pdf"
    sample_id = f"{query_idx}_{doc_name}"

    n_pages = _render_pages(pdf_path, doc_name, extract_dir, max_pages=max_pages)

    k = min(top_k, n_pages)
    page_indices: list[int] | None = None
    retrieval_mode = "first_k"
    if query_text.strip():
        ranked = _bm25_top_k(extract_dir, doc_name, n_pages, query_text, k)
        if ranked is not None:
            page_indices = ranked
            retrieval_mode = "bm25"
    if page_indices is None:
        page_indices = list(range(k))

    _write_sample(
        data_dir=data_dir,
        sample_id=sample_id,
        doc_id=sanitized_doc_id,
        query_text=query_text,
        page_indices=page_indices,
    )

    return {
        "sample_id": sample_id,
        "doc_name": doc_name,
        "n_pages": n_pages,
        "data_dir": data_dir,
        "extract_path": extract_dir,
        "retrieval_mode": retrieval_mode,
        "retrieved_pages": page_indices,
    }


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Write to a sibling .tmp file then os.replace into place.

    The extract_dir is shared across concurrent ``run_qa`` calls; two runs that
    both see "file missing" must not corrupt each other's output. ``os.replace``
    is atomic on POSIX, so the worst case is two identical writes — the second
    rename just wins, the bytes are byte-identical (deterministic render).
    """
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_text(target: Path, text: str) -> None:
    _atomic_write_bytes(target, text.encode("utf-8"))


def _render_pages(
    pdf_path: Path,
    doc_name: str,
    extract_path: Path,
    max_pages: int | None,
) -> int:
    """Render PDF pages to PNG + extract per-page text via pymupdf.

    Returns the number of pages rendered. Existing files are not re-rendered
    so repeated calls for the same doc are cheap. Writes are atomic
    (.tmp + os.replace) so concurrent runs on the same doc can't leave
    truncated PNGs / TXTs on disk.
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
                _atomic_write_bytes(img_file, pix.tobytes("png"))
            txt_file = extract_path / f"{doc_name}_{page_idx}.txt"
            if not txt_file.exists():
                _atomic_write_text(txt_file, page.get_text("text"))
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
    page_indices: list[int],
) -> dict[str, Any]:
    record = _build_sample_record(sample_id, doc_id, query_text)
    record[_R_TEXT_KEY] = page_indices
    # Image retrieval is intentionally empty — image_agent runs through NoOpModel
    # so no vision LLM is called. Setting [] (not page_indices) avoids implying
    # we have an image retrieval signal we don't actually have.
    record[_R_IMAGE_KEY] = []
    return record


def _write_sample(
    data_dir: Path,
    sample_id: str,
    doc_id: str,
    query_text: str,
    page_indices: list[int],
) -> None:
    """Overwrite ``data_dir`` with exactly one sample.

    ``data_dir`` is per-call, so we don't preserve any prior contents.
    ``page_indices`` lists which page indices to expose to both the text and
    image retrieval keys (already truncated to top-K by the caller).
    """
    samples_path = data_dir / "samples.json"
    retrieval_path = data_dir / "sample-with-retrieval-results.json"
    _save_json_list(samples_path, [_build_sample_record(sample_id, doc_id, query_text)])
    _save_json_list(
        retrieval_path,
        [_build_retrieval_record(sample_id, doc_id, query_text, page_indices)],
    )
