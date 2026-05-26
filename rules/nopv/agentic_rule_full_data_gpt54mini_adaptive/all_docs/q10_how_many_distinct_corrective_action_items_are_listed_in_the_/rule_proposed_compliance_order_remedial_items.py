import re


_HEADING_RE = re.compile(r"^\ufeff*\f*PROPOSED COMPLIANCE ORDER\s*$", re.IGNORECASE)
_REPORT_RE = re.compile(
    r"(?:it is requested\b|should any of the above\b|if any of the above\b|"
    r"follow[- ]up reports?\b|progress\s+or\s+completion\b|"
    r"outstanding\s+work\s+necessary\s+to\s+implement\b|"
    r"provide\s+documents\s+of\s+all\s+items\s+listed\s+in\s+above\b|"
    r"maintain\s+documentation\s+of\s+the\s+safety\s+improvement\s+costs\b|"
    r"submit\s+the\s+total\b|these\s+costs\s+be\s+reported\b)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\ufeff", "").replace("\f", "")).strip()


def _is_marker(text: str) -> bool:
    s = _norm(text)
    if not s:
        return False
    return bool(
        re.match(r"^(?:[A-Z](?:\.)?|[0-9]+\.(?!\d)|[ivxlcdm]+\.?|[•*-])(?:\s+.*)?$", s, re.IGNORECASE)
        or re.match(r"^\(\s*(?:[A-Z]|[0-9]+|[ivxlcdm]+)\s*\)(?:\s+.*)?$", s, re.IGNORECASE)
    )


def _is_top_level_marker(text: str) -> bool:
    s = _norm(text)
    if not s:
        return False
    return bool(
        re.match(r"^[A-Z](?:\.)?(?:\s+.*)?$", s)
        or re.match(r"^[0-9]+\.(?!\d)(?:\s+.*)?$", s)
    )


def _is_nested_marker(text: str) -> bool:
    s = _norm(text)
    if not s:
        return False
    return bool(
        re.match(r"^\(\s*(?:[A-Z]|[0-9]+|[ivxlcdm]+)\s*\)(?:\s+.*)?$", s, re.IGNORECASE)
        or re.match(r"^[ivxlcdm]+\.?(?:\s+.*)?$", s, re.IGNORECASE)
        or re.match(r"^[•*-](?:\s+.*)?$", s)
    )


def _marker_token(text: str) -> str:
    s = _norm(text)
    m = re.match(r"^([A-Z])(?:\.|\b)", s)
    if m:
        return f"{m.group(1)}."
    m = re.match(r"^([0-9]+)\.(?!\d)", s)
    if m:
        return f"{m.group(1)}."
    return s.split()[0] if s else ""


def rule_proposed_compliance_order_remedial_items(doc: dict) -> list[dict]:
    try:
        def add_span(source_line: dict, text_lines: list[str]) -> dict:
            span = {"text": "\n".join(text_lines)}
            for key in ("page_no", "paragraph_no", "line_no"):
                value = source_line.get(key)
                if value is not None:
                    span[key] = value
            return span

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
            if paragraphs:
                lines = [
                    {
                        "text": item.get("text") or "",
                        "page_no": item.get("page_no"),
                        "paragraph_no": item.get("paragraph_no"),
                    }
                    for item in paragraphs
                ]
            else:
                raw_text = doc.get("text") or ""
                if not raw_text:
                    return []
                lines = [{"text": raw_text, "page_no": None, "line_no": None, "paragraph_no": None}]

        heading_indices = [i for i, item in enumerate(lines) if _HEADING_RE.match(_norm(item.get("text") or ""))]
        if not heading_indices:
            return []
        heading_idx = heading_indices[-1]

        marker_indices = [i for i in range(heading_idx + 1, len(lines)) if _is_top_level_marker(lines[i].get("text") or "")]
        if not marker_indices:
            return []

        kept_text_lines = []
        first_span_start = None
        for pos, start_idx in enumerate(marker_indices):
            end_idx = marker_indices[pos + 1] if pos + 1 < len(marker_indices) else len(lines)

            # Top-level item blocks may contain nested roman-numeral or bullet subitems.
            # Truncate at the first nested marker so the span stays focused on the item header
            # and its main remedial sentence.
            nested_cut = None
            for i in range(start_idx + 1, end_idx):
                if _is_nested_marker(lines[i].get("text") or ""):
                    nested_cut = i
                    break

            cut_idx = nested_cut if nested_cut is not None else end_idx
            block_lines = [lines[i].get("text") or "" for i in range(start_idx, cut_idx)]
            if not block_lines:
                continue

            # Find the first report/cost boilerplate line inside the block.
            report_pos = None
            for rel_idx, raw in enumerate(block_lines):
                if _REPORT_RE.search(_norm(raw)):
                    report_pos = rel_idx
                    break

            if report_pos is not None:
                trimmed = block_lines[:report_pos]
                trimmed_nonblank = [item for item in trimmed if _norm(item)]
                if not trimmed_nonblank:
                    if kept_text_lines:
                        break
                    continue
                if len(trimmed_nonblank) == 1:
                    lone = _norm(trimmed_nonblank[0])
                    if _REPORT_RE.search(lone) or re.fullmatch(r"^(?:[A-Z](?:\.)?|[0-9]+\.(?!\d))$", lone):
                        if kept_text_lines:
                            break
                        continue
                block_lines = trimmed

            text_lines = [item for item in block_lines if _norm(item)]
            if not text_lines:
                continue

            # Skip bare marker-only blocks, which are usually the requested-cost boilerplate.
            lone_text = _norm(" ".join(text_lines))
            if re.fullmatch(r"^(?:[A-Z](?:\.)?|[0-9]+\.(?!\d))$", lone_text) and kept_text_lines:
                break

            if first_span_start is None:
                first_span_start = start_idx
            if kept_text_lines:
                kept_text_lines.append("")
            kept_text_lines.extend(text_lines)

        if not kept_text_lines or first_span_start is None:
            return []

        return [add_span(lines[first_span_start], kept_text_lines)]
    except Exception:
        return []
