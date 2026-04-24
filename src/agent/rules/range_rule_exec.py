"""Execute 5-mode range rules against markdown or normalized documents.

Modes: after, before, around, between, page.
Anchor-based modes may also use boundary_context_chars to expand beyond the
primary boundary without changing the core mode semantics.
`after` / `before` / `around` remain max_chars-bounded. `between` / `page`
use exact boundary semantics and do not silently truncate to max_chars.
All anchor modes accept an optional page_idx constraint (page is located first,
then anchor matching is performed within that page).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from agent.rules.range_rule_json import RangeRule, RetrievalSpec

_PAGE_MARKER_RE = re.compile(r"(?m)^(?:<!--\s*Page\s+(\d+)\s*-->|\[Page\s+(\d+)\])\s*$")
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)\s*$")
_SECTION_LINE_RE = re.compile(r"(?m)^\s*\[Section\]\s+(.+?)\s*$")
_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}[ \t]+")
_SECTION_PREFIX_RE = re.compile(r"^\s*\[Section\]\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_SEC_ITEM_SPLIT_RE = re.compile(r"\b(\d+)\s+([a-z])\b")
_MAX_REGEX_PATTERN_LENGTH = 120
_MAX_REGEX_MATCHES = 64
_MAX_REGEX_TOTAL_CHARS = 3000


# ---- Data structures ----


@dataclass(slots=True, frozen=True)
class RetrievedSpan:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid span bounds")
        if not self.text:
            raise ValueError("RetrievedSpan.text must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RetrievedSubset:
    matched: bool
    spans: tuple[RetrievedSpan, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "spans": [span.to_dict() for span in self.spans],
            "metadata": self.metadata,
        }


# ---- Anchor matching (three-level fallback) ----


@dataclass(slots=True, frozen=True)
class _AnchorMatch:
    start: int
    end: int
    source: str


@dataclass(slots=True, frozen=True)
class _AnchorCandidate:
    start: int
    end: int
    source: str
    normalized_text: str


@lru_cache(maxsize=4096)
def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching: strip heading prefix, casefold, collapse non-alphanumerics."""
    stripped = _HEADING_PREFIX_RE.sub("", text.strip())
    stripped = _SECTION_PREFIX_RE.sub("", stripped)
    normalized = _NON_ALNUM_RE.sub(" ", stripped.casefold()).strip()
    normalized = _SEC_ITEM_SPLIT_RE.sub(r"\1\2", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _build_anchor_candidates_uncached(text: str) -> tuple[_AnchorCandidate, ...]:
    candidates: list[_AnchorCandidate] = []

    for match in _SECTION_LINE_RE.finditer(text):
        candidates.append(
            _AnchorCandidate(
                start=match.start(),
                end=match.end(),
                source="section_line",
                normalized_text=_normalize_text(match.group(1)),
            )
        )

    for match in _MARKDOWN_HEADING_RE.finditer(text):
        candidates.append(
            _AnchorCandidate(
                start=match.start(),
                end=match.end(),
                source="markdown_heading",
                normalized_text=_normalize_text(match.group(2)),
            )
        )

    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        if line:
            candidates.append(
                _AnchorCandidate(
                    start=offset,
                    end=offset + len(line),
                    source="normalized_line",
                    normalized_text=_normalize_text(line),
                )
            )
        offset += len(raw_line)

    return tuple(candidates)


@lru_cache(maxsize=16)
def _build_anchor_candidates(text: str) -> tuple[_AnchorCandidate, ...]:
    return _build_anchor_candidates_uncached(text)


def _find_anchor(text: str, anchor: str) -> _AnchorMatch | None:
    """Three-level fallback: section/heading line → normalized line → case-insensitive substring."""
    norm_anchor = _normalize_text(anchor)
    if not norm_anchor:
        return None

    for candidate in _build_anchor_candidates(text):
        if candidate.normalized_text == norm_anchor:
            return _AnchorMatch(
                start=candidate.start,
                end=candidate.end,
                source=candidate.source,
            )

    # 3) case-insensitive substring
    idx = text.lower().find(anchor.lower())
    if idx != -1:
        return _AnchorMatch(start=idx, end=idx + len(anchor), source="substring")

    return None


# ---- Page utilities ----


def _page_boundaries(text: str) -> list[tuple[int, int, int]]:
    """Return [(page_num, start, end), ...]."""
    matches = list(_PAGE_MARKER_RE.finditer(text))
    boundaries: list[tuple[int, int, int]] = []
    for idx, m in enumerate(matches):
        page_num = int(m.group(1) or m.group(2))
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        boundaries.append((page_num, start, end))
    return boundaries


def _restrict_to_page(text: str, page_idx: int) -> tuple[str, int] | None:
    """Return (page_text, global_offset) or None if page not found."""
    for page_num, start, end in _page_boundaries(text):
        if page_num == page_idx:
            return text[start:end], start
    return None


# ---- Span construction ----


def _clip_span(text: str, start: int, end: int) -> RetrievedSpan:
    s = max(0, start)
    e = min(len(text), end)
    if e <= s:
        raise ValueError("empty span")
    return RetrievedSpan(start=s, end=e, text=text[s:e])


def _unmatched(reason: str) -> RetrievedSubset:
    return RetrievedSubset(matched=False, spans=(), metadata={"reason": reason})


def _metadata_with_context(mode: str, **metadata: Any) -> dict[str, Any]:
    boundary_context_chars = metadata.pop("boundary_context_chars", None)
    result = {"mode": mode, **metadata}
    if boundary_context_chars:
        result["boundary_context_chars"] = boundary_context_chars
    return result


# ---- Mode executors ----


def _exec_after(spec: RetrievalSpec, text: str, offset: int = 0) -> RetrievedSubset:
    anchor_match = _find_anchor(text, spec.anchor or "")
    if anchor_match is None:
        return _unmatched("anchor_not_found")
    start = max(0, anchor_match.start - spec.boundary_context_chars)
    end = min(len(text), anchor_match.start + spec.max_chars)
    span = _clip_span(text, start, end)
    # Convert back to global coordinates.
    global_span = RetrievedSpan(
        start=span.start + offset, end=span.end + offset, text=span.text
    )
    return RetrievedSubset(
        matched=True,
        spans=(global_span,),
        metadata=_metadata_with_context(
            "after",
            anchor_source=anchor_match.source,
            boundary_context_chars=spec.boundary_context_chars,
        ),
    )


def _exec_before(spec: RetrievalSpec, text: str, offset: int = 0) -> RetrievedSubset:
    anchor_match = _find_anchor(text, spec.anchor or "")
    if anchor_match is None:
        return _unmatched("anchor_not_found")
    end = min(len(text), anchor_match.end + spec.boundary_context_chars)
    start = max(0, anchor_match.end - spec.max_chars)
    span = _clip_span(text, start, end)
    global_span = RetrievedSpan(
        start=span.start + offset, end=span.end + offset, text=span.text
    )
    return RetrievedSubset(
        matched=True,
        spans=(global_span,),
        metadata=_metadata_with_context(
            "before",
            anchor_source=anchor_match.source,
            boundary_context_chars=spec.boundary_context_chars,
        ),
    )


def _exec_around(spec: RetrievalSpec, text: str, offset: int = 0) -> RetrievedSubset:
    anchor_match = _find_anchor(text, spec.anchor or "")
    if anchor_match is None:
        return _unmatched("anchor_not_found")
    center = (anchor_match.start + anchor_match.end) // 2
    half = spec.max_chars // 2
    start = max(0, center - half)
    end = min(len(text), start + spec.max_chars)
    span = _clip_span(text, start, end)
    global_span = RetrievedSpan(
        start=span.start + offset, end=span.end + offset, text=span.text
    )
    return RetrievedSubset(
        matched=True,
        spans=(global_span,),
        metadata=_metadata_with_context(
            "around",
            anchor_source=anchor_match.source,
            boundary_context_chars=spec.boundary_context_chars,
        ),
    )


def _exec_between(spec: RetrievalSpec, text: str, offset: int = 0) -> RetrievedSubset:
    match_a = _find_anchor(text, spec.anchor or "")
    match_b = _find_anchor(text, spec.anchor_b or "")
    if match_a is None or match_b is None:
        return _unmatched("anchor_pair_not_found")
    # Ensure A comes before B.
    if match_b.start <= match_a.start:
        return _unmatched("anchor_pair_not_found")
    core_start = match_a.start
    core_end = match_b.end
    start = max(0, core_start - spec.boundary_context_chars)
    end = min(len(text), core_end + spec.boundary_context_chars)
    span = _clip_span(text, start, end)
    global_span = RetrievedSpan(
        start=span.start + offset, end=span.end + offset, text=span.text
    )
    return RetrievedSubset(
        matched=True,
        spans=(global_span,),
        metadata=_metadata_with_context(
            "between",
            anchor_a_source=match_a.source,
            anchor_b_source=match_b.source,
            boundary_context_chars=spec.boundary_context_chars,
            max_chars_ignored_for_mode=True,
        ),
    )


def _exec_page(spec: RetrievalSpec, text: str) -> RetrievedSubset:
    result = _restrict_to_page(text, spec.page_idx or 0)
    if result is None:
        return _unmatched("page_not_found")
    page_text, global_offset = result
    span = _clip_span(page_text, 0, len(page_text))
    global_span = RetrievedSpan(
        start=span.start + global_offset,
        end=span.end + global_offset,
        text=span.text,
    )
    return RetrievedSubset(
        matched=True,
        spans=(global_span,),
        metadata={
            "mode": "page",
            "page_idx": spec.page_idx,
            "max_chars_ignored_for_mode": True,
        },
    )


def _is_regex_quantifier_start(pattern: str, index: int) -> bool:
    return pattern[index] in "*+?{"


def _is_supported_regex_pattern(pattern: str) -> bool:
    """Conservatively reject complex regexes to reduce ReDoS risk."""
    if not pattern or len(pattern) > _MAX_REGEX_PATTERN_LENGTH:
        return False
    if "(?" in pattern:
        return False
    if re.search(r"\\[1-9]", pattern) or "\\g<" in pattern:
        return False

    stack: list[bool] = []
    escaped = False
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char == "(":
            stack.append(False)
            index += 1
            continue
        if char == ")":
            if not stack:
                return False
            had_inner_quantifier = stack.pop()
            probe = index + 1
            while probe < len(pattern) and pattern[probe].isspace():
                probe += 1
            if had_inner_quantifier and probe < len(pattern):
                if _is_regex_quantifier_start(pattern, probe):
                    return False
            index += 1
            continue
        if stack and _is_regex_quantifier_start(pattern, index):
            stack[-1] = True
            if char == "{":
                closing = pattern.find("}", index + 1)
                if closing == -1:
                    return False
                index = closing + 1
                continue
        index += 1

    return not stack and not escaped


def _exec_regex(spec: RetrievalSpec, text: str, offset: int = 0) -> RetrievedSubset:
    """mode='regex': anchor is used as a regex pattern; returns all match spans (each capped at max_chars)."""
    pattern = spec.anchor or ""
    if not _is_supported_regex_pattern(pattern):
        return RetrievedSubset(
            matched=False,
            spans=(),
            metadata={
                "mode": "regex",
                "reason": "regex_too_complex",
                "pattern": pattern,
            },
        )
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return RetrievedSubset(
            matched=False,
            spans=(),
            metadata={"mode": "regex", "reason": "regex_error", "error": str(exc)},
        )
    spans: list[RetrievedSpan] = []
    returned_chars = 0
    truncated = False
    for m in regex.finditer(text):
        if m.start() == m.end():
            continue  # skip zero-width matches to avoid pathological results
        if len(spans) >= _MAX_REGEX_MATCHES or returned_chars >= _MAX_REGEX_TOTAL_CHARS:
            truncated = True
            break
        start = m.start()
        end = min(m.end(), start + spec.max_chars)
        remaining_chars = _MAX_REGEX_TOTAL_CHARS - returned_chars
        end = min(end, start + remaining_chars)
        if end <= start:
            truncated = True
            continue
        spans.append(
            RetrievedSpan(
                start=start + offset,
                end=end + offset,
                text=text[start:end],
            )
        )
        returned_chars += end - start
    if not spans:
        return RetrievedSubset(
            matched=False,
            spans=(),
            metadata={"mode": "regex", "reason": "regex_no_match", "pattern": pattern},
        )
    return RetrievedSubset(
        matched=True,
        spans=tuple(spans),
        metadata={
            "mode": "regex",
            "pattern": pattern,
            "match_count": len(spans),
            "returned_char_count": returned_chars,
            "truncated": truncated,
        },
    )


# ---- Dispatch table ----

_MODE_DISPATCH = {
    "after": _exec_after,
    "before": _exec_before,
    "around": _exec_around,
    "between": _exec_between,
    "regex": _exec_regex,
}


def execute_retrieval_spec(spec: RetrievalSpec, document_text: str) -> RetrievedSubset:
    """Execute a single retrieval spec against document_text."""
    if spec.mode == "page":
        return _exec_page(spec, document_text)

    executor = _MODE_DISPATCH.get(spec.mode)
    if executor is None:
        raise ValueError(f"unsupported mode: {spec.mode}")

    # Optional page_idx constraint: locate the page first, then do anchor matching within it.
    if spec.page_idx is not None:
        result = _restrict_to_page(document_text, spec.page_idx)
        if result is None:
            return _unmatched("page_not_found")
        page_text, global_offset = result
        return executor(spec, page_text, global_offset)

    return executor(spec, document_text)


def execute_range_rule(rule: RangeRule, document_text: str) -> RetrievedSubset:
    return execute_retrieval_spec(rule.retrieval_spec, document_text)


__all__ = [
    "RetrievedSpan",
    "RetrievedSubset",
    "execute_range_rule",
    "execute_retrieval_spec",
]
