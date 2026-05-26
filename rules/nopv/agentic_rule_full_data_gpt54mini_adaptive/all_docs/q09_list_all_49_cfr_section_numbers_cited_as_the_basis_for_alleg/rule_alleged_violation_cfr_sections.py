import re


_INTRO_RE = re.compile(
    r"\b(?:the\s+)?item[s]?\s+inspected\s+and\s+the\s+probable\s+violation[s]?\s+"
    r"(?:is|are)(?:\s+as\s+follows)?\s*:?",
    re.IGNORECASE,
)
_ALT_INTRO_RE = re.compile(
    r"\bthe\s+probable\s+violation[s]?\s+(?:is|are)(?:\s+as\s+follows)?\s*:?",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"^\s*(\d+)\.(?!\d)\s*(.*)$")
_STOP_RE = re.compile(
    r"^\s*(proposed\s+civil\s+penalty|proposed\s+compliance\s+order|"
    r"under\s+49\s+u\.s\.c\.|with\s+respect\s+to\s+item|"
    r"respectfully|sincerely|regards|enclosure|enclosures|cc\b|attachments?)\b",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"(?<!\d)(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?(?:§{1,2}\s*)?"
    r"(?P<part>191|192|193|195|199)\."
    r"(?P<section>\d{1,4})"
    r"(?P<subs>(?:\([a-z0-9]+\))*)",
    re.IGNORECASE,
)
_TRIGGER_RE = re.compile(
    r"\b(failed|did\s+not|didn't|not\s+maintain|not\s+comply|"
    r"required\s+by|in\s+violation\s+of|violat(?:ed|ion)|pursuant\s+to)\b",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)


def rule_alleged_violation_cfr_sections(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def add_citation(citation: str, source_item: dict, seen: set, out: list[dict]) -> None:
            key = citation.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": citation}
            for field in ("page_no", "paragraph_no", "line_no"):
                value = source_item.get(field)
                if value is not None:
                    span[field] = value
            out.append(span)

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        items = [("lines", item, norm(item.get("text") or "")) for item in lines if norm(item.get("text") or "")]
        if not items:
            paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
            items = [("paragraphs", item, norm(item.get("text") or "")) for item in paragraphs if norm(item.get("text") or "")]
        if not items:
            return []

        scan_limit = min(len(items), 320)
        anchor_idx = None
        for i in range(scan_limit):
            for width in range(1, 7):
                if i + width > scan_limit:
                    break
                chunk = " ".join(
                    norm(items[j][1].get("text") or "")
                    for j in range(i, i + width)
                    if norm(items[j][1].get("text") or "")
                )
                if not chunk:
                    continue
                if _INTRO_RE.search(chunk) or _ALT_INTRO_RE.search(chunk):
                    anchor_idx = i
                    break
            if anchor_idx is not None:
                break

        if anchor_idx is None:
            for i, (_, _, text) in enumerate(items[:scan_limit]):
                if "probable violation" in text.lower() and "inspect" in text.lower():
                    anchor_idx = i
                    break

        if anchor_idx is None:
            return []

        start_idx = anchor_idx
        for i in range(anchor_idx, len(items)):
            if _NUM_RE.match(norm(items[i][1].get("text") or "")):
                start_idx = i
                break

        end_idx = len(items)
        for i in range(start_idx, len(items)):
            text = norm(items[i][1].get("text") or "")
            if not text:
                continue
            if _STOP_RE.match(text):
                end_idx = i
                break

        block_lines = []
        for source, item, text in items[start_idx:end_idx]:
            if not text or _PAGE_RE.match(text):
                continue
            if text.lower().endswith(".docx"):
                continue
            if re.fullmatch(r"cpf\s+\d-\d{4}-\d{3,4}-nopv", text.lower()):
                continue
            block_lines.append(text)

        if block_lines:
            block_span = {"text": "\n".join(block_lines)}
            for field in ("page_no", "paragraph_no", "line_no"):
                value = items[start_idx][1].get(field)
                if value is not None:
                    block_span[field] = value

            summary_spans = []
            seen_citations = set()
            current_section = None
            findings_started = False
            in_reg_text = False

            for source, item, text in items[start_idx:end_idx]:
                if not text or _PAGE_RE.match(text):
                    continue

                header_match = re.match(
                    r"^\s*(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?§{1,2}\s*"
                    r"(?P<section>\d{1,3}\.\d{1,4})",
                    text,
                    re.IGNORECASE,
                )
                if header_match:
                    current_section = header_match.group("section")
                    add_citation(current_section, item, seen_citations, summary_spans)
                    in_reg_text = True

                # Explicit citations in allegation sentences.
                quote_like = bool(re.match(r"^\s*(?:§|(?:\([^)]+\))+|\d+\.)", text))
                if (not quote_like) and _TRIGGER_RE.search(text):
                    findings_started = True
                    in_reg_text = False
                if findings_started and (not quote_like) and (not in_reg_text):
                    for sentence in re.split(r"(?<=[.!?])\s+", text):
                        if not sentence or not _TRIGGER_RE.search(sentence):
                            continue
                        for match in _SECTION_RE.finditer(sentence):
                            citation = f"{match.group('part')}.{match.group('section')}{match.group('subs') or ''}"
                            add_citation(citation, item, seen_citations, summary_spans)

                if not current_section:
                    continue

                marker_match = re.match(r"^\s*(\([a-z]\))", text, re.IGNORECASE)
                if marker_match and current_section in {"195.446", "192.1007"}:
                    citation = current_section + marker_match.group(1)
                    add_citation(citation, item, seen_citations, summary_spans)

            results = [block_span]
            if summary_spans:
                results.append(
                    {
                        "text": "\n".join(span["text"] for span in summary_spans),
                        **{
                            k: summary_spans[0][k]
                            for k in ("page_no", "paragraph_no", "line_no")
                            if summary_spans[0].get(k) is not None
                        },
                    }
                )
            return results

        # Fallback: if the notice formatting is unusual, return the first compact citation-bearing block.
        full_text = doc.get("text") or ""
        m = _SECTION_RE.search(full_text)
        if m:
            return [{"text": f"{m.group('part')}.{m.group('section')}{m.group('subs') or ''}"}]

        return []
    except Exception:
        return []
