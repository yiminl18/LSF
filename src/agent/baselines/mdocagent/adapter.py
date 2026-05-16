"""Adapter: convert DocInputs to MDocAgent's Hydra dataset layout.

MDocAgent's ``BaseDataset`` (mydatasets/base_dataset.py) expects:
  extract_path/<doc_name>_<page_idx>.png   -- 0-indexed, PNG
  extract_path/<doc_name>_<page_idx>.txt   -- 0-indexed, per-page text
  data_dir/samples.json                    -- list of sample dicts
  data_dir/sample-with-retrieval-results.json  -- same dicts + retrieval keys

Where (from config/dataset/base.yaml):
  data_dir     = ./data/<dataset.name>      (relative to upstream/MDocAgent/)
  extract_path = ./tmp/<dataset.name>       (relative to upstream/MDocAgent/)

Retrieval keys (from config/retrieval/base.yaml, text retrieval):
  r_text_key  = "text-top-10-question"
  r_image_key = "image-top-10-question"

Values are lists of INTEGER page indices (0-indexed), matching the page
files.

RETRIEVAL STRATEGY (trivial, capped at 10 pages):
The paper uses ColBERT-style retrieval. We bypass ColBERT by pre-supplying
page indices directly. To match the key names ("top-10") and avoid context
overflow in the 5-agent pipeline (10-K filings can be 80-200 pages), we cap
the supplied set at the first 10 pages. This is a conservative but safe
default; callers can override via ``_R_MAX_PAGES``. The key names use "10"
because they mirror the top_k=10 config in config/retrieval/base.yaml.

DEVIATION from paper: ColBERT retrieval is replaced by first-N-pages
selection. The 5-agent reasoning pipeline is otherwise unchanged.

Usage as standalone prep step:
    PYTHONPATH=src python -m agent.baselines.mdocagent.adapter \
        --config src/agent/config_pdfs_10doc.yaml \
        --query 0 --doc-id AMAZON_2015_10K
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent.baselines.base import DocInputs

# Upstream MDocAgent root (git submodule)
_UPSTREAM_DIR = Path(__file__).parent / "upstream" / "MDocAgent"

# Default dataset name (matches lsf.yaml we generate)
_DEFAULT_DATASET_NAME = "lsf"

# Render DPI for page images (matching upstream's default of 144 dpi for good quality)
_RENDER_DPI = 144

# Retrieval key names (must match config/retrieval/base.yaml + text.yaml)
_R_TEXT_KEY = "text-top-10-question"
_R_IMAGE_KEY = "image-top-10-question"

# Hard cap on pages supplied to MDocAgent's text+image agents.
# Keys are named "top-10" matching config/retrieval/base.yaml top_k=10.
# Without a cap, 10-K filings (80-200 pages) would exhaust LLM context windows.
_R_MAX_PAGES = 10


def _upstream_data_dir(dataset_name: str) -> Path:
    """Return upstream/MDocAgent/data/<dataset_name>/ (created if missing)."""
    d = _UPSTREAM_DIR / "data" / dataset_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _upstream_extract_dir(dataset_name: str) -> Path:
    """Return upstream/MDocAgent/tmp/<dataset_name>/ (created if missing)."""
    d = _UPSTREAM_DIR / "tmp" / dataset_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _doc_name_from_doc_id(doc_id: str) -> str:
    """Match BaseDataset.EXTRACT_DOCUMENT_ID: strip .pdf suffix and take last path component."""
    import re
    return re.sub(r"\.pdf$", "", doc_id).split("/")[-1]


def prepare_inputs(
    doc_inputs: DocInputs,
    doc_id: str,
    dataset_name: str = _DEFAULT_DATASET_NAME,
    query_idx: int = 0,
    query_text: str = "",
    max_pages: int | None = None,
) -> dict[str, Any]:
    """Materialise one (query, doc) sample in MDocAgent's expected layout.

    Steps:
    1. Render PDF pages to PNG + extract per-page text into extract_path/.
    2. Append/update samples.json with one record for this (query_idx, doc_id).
    3. Write sample-with-retrieval-results.json with trivial retrieval
       (all page indices supplied as the retrieved set).

    Returns a dict with:
        "sample_id":    the sample["id"] string used in samples.json
        "doc_name":     the doc_name string used for page files
        "n_pages":      number of pages rendered
        "data_dir":     Path to upstream/MDocAgent/data/<dataset_name>/
        "extract_path": Path to upstream/MDocAgent/tmp/<dataset_name>/
    """
    data_dir = _upstream_data_dir(dataset_name)
    extract_path = _upstream_extract_dir(dataset_name)

    doc_name = _doc_name_from_doc_id(doc_id)
    sample_id = f"{query_idx}_{doc_name}"

    # Step 1: render pages
    n_pages = _render_pages(doc_inputs, doc_name, extract_path, max_pages=max_pages)

    # Step 2 & 3: update samples + retrieval JSON
    _upsert_sample(
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
        "extract_path": extract_path,
    }


def _render_pages(
    doc_inputs: DocInputs,
    doc_name: str,
    extract_path: Path,
    max_pages: int | None = None,
) -> int:
    """Render PDF pages to PNG and extract per-page text.

    Output file naming: <doc_name>_<page_idx>.png / <doc_name>_<page_idx>.txt
    (0-indexed, matching BaseDataset.IM_FILE / TEXT_FILE).

    Falls back to writing the full normalized text as page 0 text if PDF is
    not accessible (text-only graceful degradation).

    Returns the number of pages rendered.
    """
    if not doc_inputs.pdf_path.exists():
        # Graceful fallback: write full text as page 0 only
        txt_file = extract_path / f"{doc_name}_0.txt"
        if not txt_file.exists():
            txt_file.write_text(doc_inputs.normalized_text[:100000], encoding="utf-8")
        return 1

    try:
        import pymupdf  # type: ignore[import]
    except ImportError:
        # pymupdf not available — try pypdfium2
        return _render_pages_pypdfium2(doc_inputs, doc_name, extract_path, max_pages=max_pages)

    n_pages = 0
    try:
        with pymupdf.open(str(doc_inputs.pdf_path)) as pdf:
            effective_pages = min(len(pdf), max_pages) if max_pages is not None else len(pdf)
            for page_idx, page in enumerate(pdf):
                if page_idx >= effective_pages:
                    break
                # Image
                img_file = extract_path / f"{doc_name}_{page_idx}.png"
                if not img_file.exists():
                    pix = page.get_pixmap(dpi=_RENDER_DPI)
                    pix.save(str(img_file))
                # Text
                txt_file = extract_path / f"{doc_name}_{page_idx}.txt"
                if not txt_file.exists():
                    text = page.get_text("text")
                    txt_file.write_text(text, encoding="utf-8")
                n_pages += 1
    except Exception as exc:
        print(f"[mdocagent/adapter] pymupdf render warning for {doc_name}: {exc}")
        if n_pages == 0:
            # Fallback: write full text as page 0
            txt_file = extract_path / f"{doc_name}_0.txt"
            if not txt_file.exists():
                txt_file.write_text(doc_inputs.normalized_text[:100000], encoding="utf-8")
            n_pages = 1

    return n_pages


def _render_pages_pypdfium2(
    doc_inputs: DocInputs,
    doc_name: str,
    extract_path: Path,
    max_pages: int | None = None,
) -> int:
    """Fallback page renderer using pypdfium2."""
    n_pages = 0
    try:
        import pypdfium2 as pdfium  # type: ignore[import]

        pdf = pdfium.PdfDocument(str(doc_inputs.pdf_path))
        scale = _RENDER_DPI / 72.0
        effective_pages = min(len(pdf), max_pages) if max_pages is not None else len(pdf)
        for page_idx in range(effective_pages):
            page = pdf[page_idx]
            img_file = extract_path / f"{doc_name}_{page_idx}.png"
            if not img_file.exists():
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                pil_image.save(str(img_file), format="PNG")
            txt_file = extract_path / f"{doc_name}_{page_idx}.txt"
            if not txt_file.exists():
                # pypdfium2 doesn't easily extract text; write empty stub
                txt_file.write_text("", encoding="utf-8")
            n_pages += 1
        pdf.close()
    except Exception as exc:
        print(f"[mdocagent/adapter] pypdfium2 render warning for {doc_name}: {exc}")
        if n_pages == 0:
            txt_file = extract_path / f"{doc_name}_0.txt"
            if not txt_file.exists():
                txt_file.write_text(doc_inputs.normalized_text[:100000], encoding="utf-8")
            n_pages = 1

    return n_pages


def _load_json_list(path: Path) -> list[dict]:
    """Load a JSON array from path, returning [] if the file doesn't exist."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def _save_json_list(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_sample_record(
    sample_id: str,
    doc_id: str,
    query_text: str,
) -> dict[str, Any]:
    """Build the base sample record (no retrieval keys)."""
    return {
        "id": sample_id,
        "doc_id": doc_id,
        "question": query_text,
        "answer": "",  # ground truth withheld during inference
    }


def _build_retrieval_record(
    sample_id: str,
    doc_id: str,
    query_text: str,
    n_pages: int,
) -> dict[str, Any]:
    """Build sample record with trivial-retrieval keys (first N pages, capped).

    We supply the first min(n_pages, _R_MAX_PAGES) page indices as both the
    text and image retrieved sets.  The cap matches the "top-10" key name and
    avoids overflowing LLM context windows on long documents (10-K filings
    can be 80-200 pages).  ColBERT/ColPali retrieval is bypassed entirely.
    """
    capped_page_indices = list(range(min(n_pages, _R_MAX_PAGES)))
    record = _build_sample_record(sample_id, doc_id, query_text)
    record[_R_TEXT_KEY] = capped_page_indices
    record[_R_IMAGE_KEY] = capped_page_indices
    return record


def _upsert_sample(
    data_dir: Path,
    sample_id: str,
    doc_id: str,
    query_text: str,
    n_pages: int,
) -> None:
    """Add or replace this sample in both samples.json and the retrieval JSON."""
    samples_path = data_dir / "samples.json"
    retrieval_path = data_dir / "sample-with-retrieval-results.json"

    # samples.json
    samples = _load_json_list(samples_path)
    samples = [s for s in samples if s.get("id") != sample_id]
    samples.append(_build_sample_record(sample_id, doc_id, query_text))
    _save_json_list(samples_path, samples)

    # sample-with-retrieval-results.json
    retrieval = _load_json_list(retrieval_path)
    retrieval = [s for s in retrieval if s.get("id") != sample_id]
    retrieval.append(_build_retrieval_record(sample_id, doc_id, query_text, n_pages))
    _save_json_list(retrieval_path, retrieval)


def main(argv: list[str] | None = None) -> None:
    """Standalone prep step: prepare MDocAgent inputs for one (query, doc) pair."""
    import yaml

    parser = argparse.ArgumentParser(description="Prepare MDocAgent inputs")
    parser.add_argument(
        "--config", type=Path, default=Path("src/agent/config_pdfs_10doc.yaml")
    )
    parser.add_argument("--query", type=int, required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--dataset-name", default=_DEFAULT_DATASET_NAME)
    args = parser.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Generate dataset config YAML (copies lsf.yaml into upstream submodule)
    from agent.baselines.mdocagent.dataset_config import generate_lsf_dataset_config
    cfg_path = generate_lsf_dataset_config()
    print(f"Dataset config: {cfg_path}")

    from agent.baselines.loader import build_doc_inputs
    from agent.rule_runtime.data import get_query_text
    query_text = get_query_text(config["dataset_root"], args.query)
    doc_inputs = build_doc_inputs(config, args.query, args.doc_id)

    info = prepare_inputs(
        doc_inputs,
        doc_id=args.doc_id,
        dataset_name=args.dataset_name,
        query_idx=args.query,
        query_text=query_text,
    )
    print(f"MDocAgent inputs prepared:")
    print(f"  sample_id:    {info['sample_id']}")
    print(f"  n_pages:      {info['n_pages']}")
    print(f"  data_dir:     {info['data_dir']}")
    print(f"  extract_path: {info['extract_path']}")


if __name__ == "__main__":
    main()
