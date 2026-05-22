"""Adapter: extract per-page text from PDF or parsed_json and write
MDocAgent's expected sample layout.

MDocAgent's ``BaseDataset`` (mydatasets/base_dataset.py) reads:
    extract_path/<doc_name>_<page_idx>.png   -- 0-indexed PNG (existence-checked only)
    extract_path/<doc_name>_<page_idx>.txt   -- 0-indexed per-page text
    data_dir/samples.json                    -- list of sample dicts
    data_dir/sample-with-retrieval-results.json  -- samples + retrieval keys

Input dispatch (by file suffix in ``doc_path``):
- ``.pdf``: pymupdf renders one PNG per page and ``page.get_text("text")``
  per page TXT.
- ``.json``: parsed_json schema (``data["document"]["elements"]`` grouped by
  ``bbox[0].page_id``). We write the per-page TXT directly from that, plus a
  51-byte 1x1 placeholder PNG per page — upstream uses PNG file existence at
  ``base_dataset.py:132`` as a page-loop terminator, but the bytes are never
  loaded (image_agent runs through NoOpModel and ``disable_load_image=True``
  on the read path).

The caller passes explicit ``data_dir`` and ``extract_dir``. Convention used by
``baseline.agentic_mdocagent``:

- ``extract_dir`` is SHARED across calls (`<upstream>/tmp/lsf/`), since the
  page extracts are deterministic per ``(doc_name, page_idx)`` and benefit from
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
``baseline.mdocagent.noop_model``), so populating it would just be a
misleading duplicate of the text rank.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections import defaultdict
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

# Minimal valid 1x1 transparent PNG. Upstream's base_dataset.py:132 page loop
# terminates on the first missing ``<doc>_<page>.png`` — so the parsed_json
# extraction path writes this sentinel per page to satisfy that existence
# check. The bytes are never decoded: image_agent runs through NoOpModel and
# upstream's read path uses ``disable_load_image=True``, so ``load_image()``
# is never invoked.
_PLACEHOLDER_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _doc_name_from_doc_id(doc_id: str) -> str:
    """Strip ``.pdf`` or ``.json``, take the basename, and replace any chars
    that aren't safe in filenames or Hydra interpolation. Upstream's
    BaseDataset uses ``doc_id`` raw to build PNG/TXT paths, so this prefix
    must survive shells, POSIX filesystems, and OmegaConf ``${...}``
    evaluation."""
    raw = Path(re.sub(r"\.(pdf|json)$", "", doc_id, flags=re.IGNORECASE)).name
    return _UNSAFE_DOC_NAME_RE.sub("_", raw)


def prepare_inputs(
    doc_path: Path | str,
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
    populated with the doc's per-page extracts (cache-aware: existing files are
    not re-written). When ``query_text`` is non-empty and ``rank_bm25`` is
    importable, the retrieval keys are populated with the BM25 top-K pages
    instead of the first K; otherwise it falls back to the first K.

    ``doc_path`` may be a ``.pdf`` (pymupdf path) or a ``.json`` parsed_json
    file (officeqa-style page-element schema). See module docstring for the
    dispatch contract.
    """
    doc_path = Path(doc_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"Source not found for MDocAgent prep: {doc_path}")

    data_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    doc_name = _doc_name_from_doc_id(doc_id)
    # The samples.json doc_id is what upstream's BaseDataset.EXTRACT_DOCUMENT_ID
    # parses (regex-stripping ``.pdf``) back into the PNG/TXT filename prefix.
    # We always emit a ``.pdf`` suffix here regardless of the actual input
    # source so upstream's strip produces the right doc_name.
    sanitized_doc_id = f"{doc_name}.pdf"
    sample_id = f"{query_idx}_{doc_name}"

    suffix = doc_path.suffix.lower()
    if suffix == ".pdf":
        n_pages = _extract_pages_from_pdf(doc_path, doc_name, extract_dir, max_pages=max_pages)
    elif suffix == ".json":
        n_pages = _extract_pages_from_parsed_json(doc_path, doc_name, extract_dir, max_pages=max_pages)
    else:
        raise ValueError(
            f"Unsupported doc_path suffix {doc_path.suffix!r}; expected .pdf or .json"
        )

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


def _extract_pages_from_pdf(
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


def _extract_pages_from_parsed_json(
    parsed_path: Path,
    doc_name: str,
    extract_path: Path,
    max_pages: int | None,
) -> int:
    """Extract per-page text from an officeqa-style parsed_json file.

    Groups ``data["document"]["elements"]`` by ``el["bbox"][0]["page_id"]``
    (1-indexed in the source) and writes ``<doc_name>_<page_idx>.txt`` with
    0-indexed page_idx, mirroring the PDF flow's filename convention. Also
    writes a 1x1 placeholder PNG per page — see ``_PLACEHOLDER_PNG_BYTES``
    for why upstream needs the PNG to exist.

    Returns the number of pages written.
    """
    data = json.loads(parsed_path.read_text(encoding="utf-8"))
    elements = data.get("document", {}).get("elements", [])
    pages: dict[int, list[str]] = defaultdict(list)
    for el in elements:
        content = el.get("content")
        if not content:
            continue
        bbox_list = el.get("bbox") or []
        if not bbox_list:
            continue
        page_id = bbox_list[0].get("page_id")
        if page_id is None:
            continue
        pages[page_id].append(str(content))
    if not pages:
        raise RuntimeError(f"No page-tagged text extracted from {parsed_path}")

    sorted_page_ids = sorted(pages.keys())
    effective = (
        min(len(sorted_page_ids), max_pages)
        if max_pages is not None
        else len(sorted_page_ids)
    )

    n_pages = 0
    for page_idx in range(effective):
        page_id = sorted_page_ids[page_idx]
        txt_file = extract_path / f"{doc_name}_{page_idx}.txt"
        if not txt_file.exists():
            _atomic_write_text(txt_file, "\n".join(pages[page_id]))
        img_file = extract_path / f"{doc_name}_{page_idx}.png"
        if not img_file.exists():
            _atomic_write_bytes(img_file, _PLACEHOLDER_PNG_BYTES)
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
