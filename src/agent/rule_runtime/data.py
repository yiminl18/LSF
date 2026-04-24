"""Data loading and reconstructed-document text rendering for rule agents."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Heading level prefix map
_HEADING_MAP: dict[str, str] = {
    "H1": "#",
    "H2": "##",
    "H3": "###",
    "H4": "####",
    "H5": "#####",
    "H6": "######",
    "H7": "#######",
}


@dataclass
class DocumentSample:
    """Data bundle for a single document."""

    doc_id: str
    markdown_text: str
    ground_truth_answer: str
    token_count: int


@dataclass
class QueryPackage:
    """All available documents for a single query."""

    query_idx: int
    query_text: str
    documents: list[DocumentSample] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(d.token_count for d in self.documents)


def _resolve_dataset_root(config: dict[str, Any]) -> Path:
    """Resolve dataset_root to an absolute path."""
    root = Path(config["dataset_root"])
    if not root.is_absolute():
        # Relative to repo root (src/agent/rule_runtime/ -> src/agent/ -> src/ -> repo root)
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        root = repo_root / root
    return root


def _get_processing_dir(config: dict[str, Any]) -> Path:
    root = _resolve_dataset_root(config)
    parser = config.get("parser", "docling")
    if parser == "mineru":
        return root / "processing_mineru"
    return root / "processing"


def _get_label_dir(config: dict[str, Any]) -> Path:
    root = _resolve_dataset_root(config)
    parser = config.get("parser", "docling")
    if parser == "mineru":
        return root / "label_mineru"
    return root / "label"


def _get_label_filename(config: dict[str, Any], query_idx: int) -> str:
    dataset = config["dataset"]
    if dataset == "sci-docs":
        return f"sci-docs_q{query_idx}_scibench_reconstructed_labels.json"
    return f"10k_q{query_idx}_reconstructed_labels.json"


def reconstruct_to_markdown(
    reconstruct_path: str | Path,
    truncate_before: str | None = None,
) -> str:
    """
    Convert reconstructed JSON to markdown plain text.

    Conversion rules:
    - section_header with structure.level H1-H7 -> markdown heading
    - text -> plain paragraph
    - other labels -> [label] text (defensive fallback)
    - page changes insert an <!-- Page N --> comment
    - blocks separated by blank lines

    truncate_before: regex; stops conversion at the first matching section_header.
        E.g. r"item\\s*2\\b" truncates before Item 2 (useful for 10-K filings).
    """
    path = Path(reconstruct_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    blocks: list[str] = []
    current_page: int | None = None

    for entry in data.get("texts", []):
        text = entry.get("text", "").strip()
        if not text:
            continue

        label = entry.get("label", "text")
        page_no = entry.get("page_no")
        structure = entry.get("structure", {})
        level = structure.get("level", "Body")

        # Stop if this section_header matches the truncation pattern
        if (
            truncate_before
            and label == "section_header"
            and re.search(truncate_before, text, re.IGNORECASE)
        ):
            break

        # Insert page marker on page change
        if page_no is not None and page_no != current_page:
            blocks.append(f"<!-- Page {page_no} -->")
            current_page = page_no

        if label == "section_header":
            prefix = _HEADING_MAP.get(level, "")
            if prefix:
                blocks.append(f"{prefix} {text}")
            else:
                # Body level section_header → plain text
                blocks.append(text)
        elif label == "text":
            blocks.append(text)
        else:
            # Defensive fallback for other labels (caption, table, formula, etc.)
            blocks.append(f"[{label}] {text}")

    return "\n\n".join(blocks)


_SENTENCE_END_RE = re.compile(r"[.?!:;]\s*$")

# Normalize SEC filing section prefixes:
#   "ITEM 1." / "Item 1A." / "PART I." -> "Item 1" / "Item 1A" / "Part I"
#   Capitalize the keyword, strip trailing period.
_SEC_SECTION_PREFIX_RE = re.compile(
    r"^((?:Item|Part)\s+(?:\d+[A-Za-z]?|[IVX]+))\.?\s*",
    re.IGNORECASE,
)


def _normalize_section_text(text: str) -> str:
    """Normalize SEC filing prefixes in section_header text."""
    m = _SEC_SECTION_PREFIX_RE.match(text)
    if m is None:
        return text
    prefix_raw = m.group(1)  # e.g. "ITEM 1A" or "Part II"
    # Extract keyword and number+suffix
    kw_end = 4  # "Item" and "Part" are both 4 chars
    keyword = prefix_raw[:kw_end].capitalize()  # "Item" or "Part"
    num_suffix = prefix_raw[kw_end:].strip().upper()  # "1A", "II" etc.
    prefix_normalized = f"{keyword} {num_suffix}"
    rest = text[m.end() :].strip()
    if rest:
        return f"{prefix_normalized} {rest}"
    return prefix_normalized


def reconstruct_to_normalized_text(
    reconstruct_path: str | Path,
    truncate_before: str | None = None,
) -> str:
    """Convert reconstructed JSON to normalized plain text.

    Differences from reconstruct_to_markdown:
    - section_header -> ``[Section] text`` (no # level prefix)
    - Uses text_span to merge child nodes under a section (skips children already covered by text_span)
    - Consecutive text nodes without a parent: joined with a space when on the same page and
      the previous node does not end with a sentence-final punctuation mark
    - Page markers use ``[Page N]`` format
    """
    path = Path(reconstruct_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    texts = data.get("texts", [])

    # Build children_of_header: indices of all non-header nodes covered by a section_header's text_span
    children_of_header: set[int] = set()
    for idx, entry in enumerate(texts):
        if entry.get("label") != "section_header":
            continue
        if not entry.get("text_span", "").strip():
            continue
        # Direct non-header children of this header
        for child_idx, child in enumerate(texts):
            if child.get("structure", {}).get("parent_id") == idx:
                if child.get("label") != "section_header":
                    children_of_header.add(child_idx)

    blocks: list[str] = []
    current_page: int | None = None
    # Temporary buffer for merging consecutive text nodes (e.g. cover-page fields)
    pending_texts: list[str] = []
    pending_page: int | None = None

    def _flush_pending() -> None:
        if pending_texts:
            blocks.append(" ".join(pending_texts))
            pending_texts.clear()

    for idx, entry in enumerate(texts):
        text = entry.get("text", "").strip()
        if not text:
            continue

        label = entry.get("label", "text")
        page_no = entry.get("page_no")
        text_span = entry.get("text_span", "").strip()

        # Stop at the first section_header matching the truncation pattern
        if (
            truncate_before
            and label == "section_header"
            and re.search(truncate_before, text, re.IGNORECASE)
        ):
            break

        # Skip child nodes already covered by their parent header's text_span
        if idx in children_of_header:
            continue

        # Page change
        if page_no is not None and page_no != current_page:
            _flush_pending()
            blocks.append(f"[Page {page_no}]")
            current_page = page_no

        if label == "section_header":
            _flush_pending()
            normalized_heading = _normalize_section_text(text)
            if text_span:
                blocks.append(f"[Section] {normalized_heading}\n{text_span}")
            else:
                blocks.append(f"[Section] {normalized_heading}")
        elif idx not in children_of_header:
            # Not a section_header and not a covered child — may be cover-page or parentless text
            parent_id = entry.get("structure", {}).get("parent_id")
            if parent_id is not None and parent_id not in children_of_header:
                # Has a parent but the parent has no text_span (or parent is itself skipped)
                # Treat as standalone text
                pass

            # Merge consecutive text nodes on the same page when no sentence boundary
            if pending_texts and (
                page_no != pending_page or _SENTENCE_END_RE.search(pending_texts[-1])
            ):
                _flush_pending()

            pending_page = page_no
            if label == "text":
                pending_texts.append(text)
            else:
                # checkbox, table, caption, etc.
                pending_texts.append(f"[{label}] {text}")

    _flush_pending()
    return "\n\n".join(blocks)


def extract_ground_truth(
    label_json_path: str | Path,
    doc_id: str,
    query_idx: int,
) -> str:
    """
    Extract the GT answer for a doc-query pair from a label JSON file.

    - Locates the entry where doc_name == doc_id
    - Asserts question_idx == query_idx (consistency check)
    - Returns the ground_truth field
    - Raises ValueError if the doc is missing or ground_truth is empty
    """
    path = Path(label_json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for label_entry in data.get("labels", []):
        if label_entry.get("doc_name") == doc_id:
            actual_idx = label_entry.get("question_idx")
            assert actual_idx == query_idx, (
                f"question_idx mismatch: expected {query_idx}, got {actual_idx}, doc={doc_id}"
            )
            gt = label_entry.get("ground_truth", "")
            if not gt:
                raise ValueError(
                    f"doc={doc_id} query_idx={query_idx} has empty ground_truth"
                )
            return gt

    raise ValueError(f"doc_name={doc_id} not found in label file (query_idx={query_idx})")


def get_query_text(dataset_root: str | Path, query_idx: int) -> str:
    """Read line query_idx from queries.txt as the query text (authoritative source, 0-indexed)."""
    queries_path = Path(dataset_root) / "queries.txt"
    with queries_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if query_idx < 0 or query_idx >= len(lines):
        raise ValueError(
            f"query_idx={query_idx} out of range for queries.txt ({len(lines)} lines)"
        )
    return lines[query_idx]


from core.llm.tokens import estimate_tokens  # re-exported for agent callers


def build_query_package(
    config: dict[str, Any],
    query_idx: int,
    valid_docs: list[str],
    text_format: str = "markdown",
) -> QueryPackage:
    """
    Build a QueryPackage.

    Required config keys:
    - dataset_root: str (dataset root directory; relative paths accepted)
    - dataset: str (e.g. "sci-docs")
    - parser: str ("mineru" or "docling")

    valid_docs: list of doc_ids (filenames without the _reconstructed.json suffix)
    text_format: "markdown" (default) or "normalized" (normalized plain text, used for python_code mode)
    """
    dataset_root = _resolve_dataset_root(config)
    processing_dir = _get_processing_dir(config)
    label_dir = _get_label_dir(config)
    label_filename = _get_label_filename(config, query_idx)
    label_path = label_dir / label_filename

    query_text = get_query_text(dataset_root, query_idx)

    truncate_before = config.get("truncate_before")
    reconstruct_fn = (
        reconstruct_to_normalized_text
        if text_format == "normalized"
        else reconstruct_to_markdown
    )

    documents: list[DocumentSample] = []
    for doc_id in valid_docs:
        reconstruct_path = processing_dir / f"{doc_id}_reconstructed.json"
        markdown_text = reconstruct_fn(
            reconstruct_path, truncate_before=truncate_before
        )
        ground_truth_answer = extract_ground_truth(label_path, doc_id, query_idx)
        token_count = estimate_tokens(markdown_text)

        documents.append(
            DocumentSample(
                doc_id=doc_id,
                markdown_text=markdown_text,
                ground_truth_answer=ground_truth_answer,
                token_count=token_count,
            )
        )

    return QueryPackage(
        query_idx=query_idx,
        query_text=query_text,
        documents=documents,
    )
