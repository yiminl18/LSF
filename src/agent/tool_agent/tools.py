"""Tool Registry — minimal rule-validation toolset for the tool-agent."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from agent.rules.range_rule_exec import execute_range_rule
from agent.rules.range_rule_json import RangeRule, RetrievalSpec
from agent.tool_agent.document import DocumentContext

_SECTION_MARKER = "[Section] "
_PAGE_LINE_RE = re.compile(r"^\[Page\s+([^\]]+)\]\s*$")
_ANSWER_HINT_MAX_LEN = 100
_FIND_SECTION_TOP_K = 3
_FIND_SECTION_BODY_WINDOW = 500
_FIND_SECTION_PREVIEW_CHARS_DEFAULT = 1500  # default chars per section hit (top-3 × 1500 ≈ 56% obs budget)
_FIND_SECTION_PREVIEW_CHARS_MAX = 4000      # agent-specifiable upper bound
_BATCH_SPAN_PREVIEW_CHARS = 800
# get_section / get_page defaults and hard cap (leaves room for JSON framing)
_TREE_TOOL_MAX_CHARS_DEFAULT = 4000
_TREE_TOOL_MAX_CHARS_CAP = 6000

# Stopword denylist for extraction-precision lint:
# if the hint regex extracts a token that is always one of these AND identical
# across all matched docs, the pattern is likely too generic.
_HINT_LINT_STOPWORD_DENYLIST: frozenset[str] = frozenset({
    "item", "section", "the", "a", "an", "of", "in", "and", "or",
    "page", "chapter", "part", "table", "form", "exhibit",
})


@dataclass(slots=True)
class ToolResult:
    """Result of a single tool call."""

    name: str
    success: bool
    data: Any
    error: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class ToolRegistry:
    """Rule-validation tools for the tool-agent."""

    TOOL_NAMES: tuple[str, ...] = (
        "batch_apply_rule",
        "apply_rule",
        "find_section",
        "get_section",
        "get_page",
    )

    def __init__(
        self,
        doc_context: DocumentContext,
        peer_docs: list[DocumentContext] | None = None,
    ) -> None:
        self._doc = doc_context
        self._peer_docs: list[DocumentContext] = list(peer_docs) if peer_docs else []
        self._dispatch_map: dict[str, Any] = {
            "batch_apply_rule": self._tool_batch_apply_rule,
            "apply_rule": self._tool_apply_rule,
            "find_section": self._tool_find_section,
            "get_section": self._tool_get_section,
            "get_page": self._tool_get_page,
        }

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call; catches all exceptions and returns a failed ToolResult."""
        if tool_name not in self._dispatch_map:
            return ToolResult(
                name=tool_name,
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}. Available: {', '.join(self.TOOL_NAMES)}",
            )
        t0 = time.time()
        try:
            data, cost = self._dispatch_map[tool_name](args)
            elapsed = (time.time() - t0) * 1000
            return ToolResult(
                name=tool_name,
                success=True,
                data=data,
                cost_usd=cost,
                latency_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            return ToolResult(
                name=tool_name,
                success=False,
                data=None,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=elapsed,
            )

    def _resolve_doc(self, doc_id: str | None) -> DocumentContext | dict[str, Any]:
        """Resolve doc_id to a DocumentContext."""
        if doc_id is None or doc_id == self._doc.doc_id:
            return self._doc
        for peer in self._peer_docs:
            if peer.doc_id == doc_id:
                return peer
        available = [self._doc.doc_id] + [p.doc_id for p in self._peer_docs]
        return {"error": f"Unknown doc_id: {doc_id!r}. Available: {available}"}

    def _build_rule(self, rule_spec: dict[str, Any], *, rule_text: str) -> RangeRule:
        spec = RetrievalSpec(**rule_spec)
        return RangeRule(
            rule_text=rule_text,
            evidence_basis=rule_text,
            retrieval_spec=spec,
        )

    def _tool_batch_apply_rule(self, args: dict[str, Any]) -> tuple[Any, float]:
        """Apply a rule across all docs (current + peers) and return per-doc previews."""
        rule_spec = args.get("rule_spec")
        if rule_spec is None:
            return {"error": "Missing required argument: 'rule_spec'"}, 0.0
        try:
            rule = self._build_rule(rule_spec, rule_text="batch_probe")
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid rule_spec: {exc}"}, 0.0

        answer_hint_pattern = args.get("answer_hint_pattern")
        if answer_hint_pattern is not None and not isinstance(answer_hint_pattern, str):
            return {"error": "answer_hint_pattern must be a string or null"}, 0.0
        if answer_hint_pattern is not None and len(answer_hint_pattern) > _ANSWER_HINT_MAX_LEN:
            return {
                "error": (
                    f"answer_hint_pattern too long "
                    f"({len(answer_hint_pattern)} > {_ANSWER_HINT_MAX_LEN} chars)"
                )
            }, 0.0

        hint_regex: re.Pattern[str] | None = None
        if answer_hint_pattern:
            try:
                hint_regex = re.compile(answer_hint_pattern)
            except re.error as exc:
                return {"error": f"Invalid answer_hint_pattern regex: {exc}"}, 0.0

        docs = [self._doc, *self._peer_docs]
        per_doc: list[dict[str, Any]] = []
        matched = 0
        unmatched_doc_ids: list[str] = []
        answer_present_count = 0
        answer_present_doc_ids: list[str] = []
        # Anchor universality: does the anchor literal appear in each doc at all,
        # independent of whether the rule matched? Helps the agent see "anchor only
        # in 2/10 docs" early so it tries a more universal phrase instead of blaming
        # downstream coverage.
        anchor_str = (rule_spec.get("anchor") or "") if isinstance(rule_spec, dict) else ""
        anchor_lower = anchor_str.casefold().strip()
        anchor_doc_hits = 0
        # Extraction precision lint: collect first regex match per matched doc;
        # flag if all extracted tokens are identical AND in the stopword denylist
        # (e.g. a capitalised-word pattern that always matches "Item").
        extracted_tokens: list[str] = []
        for doc in docs:
            try:
                subset = execute_range_rule(rule, doc.normalized_text)
            except Exception as exc:
                per_doc.append(
                    {
                        "doc_id": doc.doc_id,
                        "matched": False,
                        "span_preview": "",
                        "char_count": 0,
                        "error": str(exc),
                    }
                )
                unmatched_doc_ids.append(doc.doc_id)
                continue

            span_text = "\n\n".join(s.text for s in subset.spans) if subset.spans else ""
            entry: dict[str, Any] = {
                "doc_id": doc.doc_id,
                "matched": bool(subset.matched),
                "span_preview": span_text[:_BATCH_SPAN_PREVIEW_CHARS],
                "char_count": len(span_text),
            }
            if anchor_lower and anchor_lower in doc.normalized_text.casefold():
                anchor_doc_hits += 1
            if subset.matched:
                matched += 1
                if hint_regex is not None:
                    m = hint_regex.search(span_text)
                    hit = m is not None
                    entry["answer_present"] = hit
                    if hit:
                        answer_present_count += 1
                        answer_present_doc_ids.append(doc.doc_id)
                        extracted_tokens.append(m.group(0).strip())
            else:
                unmatched_doc_ids.append(doc.doc_id)
            per_doc.append(entry)

        total = len(docs)
        result: dict[str, Any] = {
            "per_doc": per_doc,
            "matched_count": matched,
            "total_docs": total,
            "coverage": round(matched / total, 4) if total else 0.0,
            "unmatched_doc_ids": unmatched_doc_ids,
        }
        if anchor_lower:
            result["anchor_universality"] = {
                "docs_containing_anchor": anchor_doc_hits,
                "total_docs": total,
            }
        if hint_regex is not None:
            result["answer_present_count"] = answer_present_count
            result["answer_present_doc_ids"] = answer_present_doc_ids
            if extracted_tokens:
                unique_tokens = {t.casefold() for t in extracted_tokens}
                if (
                    len(unique_tokens) == 1
                    and len(extracted_tokens) >= 2
                    and next(iter(unique_tokens)) in _HINT_LINT_STOPWORD_DENYLIST
                ):
                    result["hint_extraction_warning"] = (
                        f"hint regex extracted only {next(iter(unique_tokens))!r} "
                        f"across {len(extracted_tokens)} matched docs — pattern likely "
                        "too generic, will fail Phase B extraction precision gate"
                    )
        return result, 0.0

    def _tool_find_section(self, args: dict[str, Any]) -> tuple[Any, float]:
        """Search a single doc for section headings or first-500-char body matching keyword; return top-3.

        args:
          keyword: required search term
          doc_id: optional target doc (defaults to primary doc)
          max_chars: optional per-hit body chars to return
            default _FIND_SECTION_PREVIEW_CHARS_DEFAULT (1500), agent may raise up to
            _FIND_SECTION_PREVIEW_CHARS_MAX (4000); TOC [Nc] tag shows total section
            length so the agent can request the full section when needed.
        """
        keyword = args.get("keyword")
        if not isinstance(keyword, str) or not keyword:
            return {"error": "Missing or empty 'keyword'"}, 0.0

        raw_mc = args.get("max_chars")
        if isinstance(raw_mc, int) and raw_mc > 0:
            max_chars = min(raw_mc, _FIND_SECTION_PREVIEW_CHARS_MAX)
        else:
            max_chars = _FIND_SECTION_PREVIEW_CHARS_DEFAULT

        doc = self._resolve_doc(args.get("doc_id"))
        if isinstance(doc, dict):
            return doc, 0.0

        sections = _parse_sections(doc.normalized_text)
        keyword_lower = keyword.casefold()

        seen: set[tuple[str, int]] = set()
        matches: list[dict[str, Any]] = []
        for section in sections:
            identity = (section["heading"], section["start"])
            if identity in seen:
                continue
            heading_hit = keyword_lower in section["heading"].casefold()
            body_window = section["body"][:_FIND_SECTION_BODY_WINDOW]
            body_hit = keyword_lower in body_window.casefold()
            if not (heading_hit or body_hit):
                continue
            seen.add(identity)
            preview_source = section["body"] if section["body"] else section["heading"]
            body_chars = len(preview_source)
            preview = preview_source[:max_chars]
            matches.append(
                {
                    "heading": section["heading"],
                    "page": section["page"],
                    "body_chars": body_chars,
                    "preview": preview,
                    "preview_truncated": body_chars > max_chars,
                }
            )

        total_matches = len(matches)
        top = matches[:_FIND_SECTION_TOP_K]
        return {
            "sections": top,
            "truncated": total_matches > _FIND_SECTION_TOP_K,
            "total_matches": total_matches,
            "max_chars_used": max_chars,
        }, 0.0

    def _tool_apply_rule(self, args: dict[str, Any]) -> tuple[Any, float]:
        """Apply a candidate RangeRule to a single doc."""
        rule_spec = args.get("rule_spec")
        if rule_spec is None:
            return {"error": "Missing required argument: 'rule_spec'"}, 0.0
        doc = self._resolve_doc(args.get("doc_id"))
        if isinstance(doc, dict):
            return doc, 0.0
        rule = self._build_rule(rule_spec, rule_text="agent_candidate")
        subset = execute_range_rule(rule, doc.normalized_text)
        span_text = "\n\n".join(s.text for s in subset.spans) if subset.spans else ""
        return {
            "matched": bool(subset.matched),
            "text": span_text[:3000],
            "char_count": len(span_text),
            "metadata": subset.metadata,
            "preview": span_text[:500],
        }, 0.0

    # ---- Structured-access tools (read directly from the reconstructed.json tree) ----

    def _tool_get_section(self, args: dict[str, Any]) -> tuple[Any, float]:
        """Return the complete body of a section and its subtree by section_id.

        args:
          section_id: int (required) — the N from `[id=N]` in the TOC
          doc_id: str (optional, defaults to primary doc)
          max_chars: int (optional, default 4000, cap 6000)
          include_tables: bool (optional, default True)
        """
        section_id = args.get("section_id")
        if not isinstance(section_id, int):
            return {"error": "Missing or non-int 'section_id'"}, 0.0

        doc = self._resolve_doc(args.get("doc_id"))
        if isinstance(doc, dict):
            return doc, 0.0

        if section_id not in doc.section_index:
            return {"error": f"section_id={section_id} not found in doc={doc.doc_id}"}, 0.0
        root_entry = doc.section_index[section_id]

        raw_mc = args.get("max_chars")
        max_chars = (
            min(int(raw_mc), _TREE_TOOL_MAX_CHARS_CAP)
            if isinstance(raw_mc, int) and raw_mc > 0
            else _TREE_TOOL_MAX_CHARS_DEFAULT
        )
        include_tables = bool(args.get("include_tables", True))

        # BFS to collect all descendant indices (including root); sorted for reading order
        descendants = _collect_descendants(section_id, doc.entries)
        ordered = sorted(descendants)

        body_parts: list[str] = []
        child_section_ids: list[int] = []
        table_ids: list[int] = []
        pages: list[int] = []
        for idx in ordered:
            entry = doc.entries[idx]
            if entry.get("page_no") is not None and entry["page_no"] not in pages:
                pages.append(entry["page_no"])
            label = entry.get("label", "")
            text = (entry.get("text") or "").strip()
            if idx == section_id:
                # Root header itself is surfaced via the heading field, not body
                continue
            if label == "section_header":
                child_section_ids.append(idx)
                body_parts.append(f"\n## [id={idx}] {text}\n")
            elif label == "list_item":
                body_parts.append(f"- {text}")
            elif label == "table":
                table_ids.append(idx)
                if include_tables:
                    td = entry.get("table_data") or {}
                    rows = td.get("num_rows", "?")
                    cols = td.get("num_cols", "?")
                    page = entry.get("page_no", "?")
                    body_parts.append(f"[Table id={idx} at p{page}, {rows}×{cols} cells]")
            elif text:
                body_parts.append(text)

        body_full = "\n".join(body_parts).strip()
        body_chars_full = len(body_full)
        body = body_full
        body_truncated = False
        if body_chars_full > max_chars:
            body = body_full[:max_chars] + f"\n...[truncated at {max_chars}/{body_chars_full}]"
            body_truncated = True

        heading = (root_entry.get("text") or "").strip()
        level = root_entry.get("structure", {}).get("level") or "H?"
        return {
            "section_id": section_id,
            "doc_id": doc.doc_id,
            "heading": heading,
            "level": level,
            "pages": pages,
            "body": body,
            "body_chars": body_chars_full,
            "body_truncated": body_truncated,
            "child_section_ids": child_section_ids,
            "table_ids_in_body": table_ids,
            "max_chars_used": max_chars,
        }, 0.0

    def _tool_get_page(self, args: dict[str, Any]) -> tuple[Any, float]:
        """Return all entries on a given page (optional label filter).

        args:
          page_idx: int (required)
          doc_id: str (optional)
          labels: list[str] (optional label filter; default = all)
          max_chars: int (optional, default 4000, cap 6000)
        """
        page_idx = args.get("page_idx")
        if not isinstance(page_idx, int):
            return {"error": "Missing or non-int 'page_idx'"}, 0.0

        doc = self._resolve_doc(args.get("doc_id"))
        if isinstance(doc, dict):
            return doc, 0.0

        labels_filter = args.get("labels")
        if labels_filter is not None and not isinstance(labels_filter, list):
            return {"error": "'labels' must be a list of label strings"}, 0.0
        allowed: set[str] | None = set(labels_filter) if labels_filter else None

        raw_mc = args.get("max_chars")
        max_chars = (
            min(int(raw_mc), _TREE_TOOL_MAX_CHARS_CAP)
            if isinstance(raw_mc, int) and raw_mc > 0
            else _TREE_TOOL_MAX_CHARS_DEFAULT
        )

        label_hist: dict[str, int] = {}
        body_parts: list[str] = []
        entries_count = 0
        for idx, entry in enumerate(doc.entries):
            if entry.get("page_no") != page_idx:
                continue
            label = entry.get("label", "")
            if allowed is not None and label not in allowed:
                continue
            entries_count += 1
            label_hist[label] = label_hist.get(label, 0) + 1
            text = (entry.get("text") or "").strip()
            if label == "section_header":
                body_parts.append(f"\n## [id={idx}] {text}\n")
            elif label == "list_item":
                body_parts.append(f"- {text}")
            elif label == "table":
                td = entry.get("table_data") or {}
                rows = td.get("num_rows", "?")
                cols = td.get("num_cols", "?")
                body_parts.append(f"[Table id={idx}, {rows}×{cols} cells]")
            elif text:
                body_parts.append(text)

        body_full = "\n".join(body_parts).strip()
        body_chars_full = len(body_full)
        body = body_full
        body_truncated = False
        if body_chars_full > max_chars:
            body = body_full[:max_chars] + f"\n...[truncated at {max_chars}/{body_chars_full}]"
            body_truncated = True

        if entries_count == 0:
            return {"error": f"no entries on page_idx={page_idx} in doc={doc.doc_id}"}, 0.0

        return {
            "doc_id": doc.doc_id,
            "page_no": page_idx,
            "entries_count": entries_count,
            "body": body,
            "body_chars": body_chars_full,
            "body_truncated": body_truncated,
            "label_histogram": label_hist,
            "max_chars_used": max_chars,
        }, 0.0


def _collect_descendants(root_id: int, entries: list[dict[str, Any]]) -> set[int]:
    """BFS along structure.parent_id from root_id; returns all descendant indices including root."""
    descendants: set[int] = {root_id}
    frontier = [root_id]
    while frontier:
        next_frontier: list[int] = []
        for pid in frontier:
            for idx, entry in enumerate(entries):
                if idx in descendants:
                    continue
                if entry.get("structure", {}).get("parent_id") == pid:
                    descendants.add(idx)
                    next_frontier.append(idx)
        frontier = next_frontier
    return descendants


def _parse_sections(normalized_text: str) -> list[dict[str, Any]]:
    """Parse sections delimited by [Section] markers, tracking the most recent [Page xxx] page number."""
    sections: list[dict[str, Any]] = []
    current_page: str | None = None
    lines = normalized_text.splitlines(keepends=True)

    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        page_match = _PAGE_LINE_RE.match(line)
        if page_match is not None:
            current_page = page_match.group(1).strip()
            continue
        if not line.startswith(_SECTION_MARKER):
            continue

        heading = line[len(_SECTION_MARKER):].strip()
        body_start = line_starts[idx] + len(raw_line)
        body_end = len(normalized_text)
        for next_idx in range(idx + 1, len(lines)):
            if lines[next_idx].startswith(_SECTION_MARKER):
                body_end = line_starts[next_idx]
                break
        body = normalized_text[body_start:body_end].strip()

        sections.append(
            {
                "heading": heading,
                "page": current_page,
                "start": line_starts[idx],
                "body": body,
            }
        )
    return sections
