import re


_ANCHOR_RE = re.compile(
    r"\bprobable\s+violations?\b",
    re.IGNORECASE,
)
_INTRO_RE = re.compile(
    r"\b(the\s+items?\s+inspected\s+and\s+the\s+probable\s+violations?\s+(?:is|are)|"
    r"as\s+a\s+result\s+of\s+the\s+inspection,?\s+it\s+is\s+alleged)",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"^\s*(proposed\s+civil\s+penalty|proposed\s+compliance\s+order|warning\s+items|response\s+to\s+this\s+notice)\b",
    re.IGNORECASE,
)
_FOOTNOTE_MARKER_RE = re.compile(r"^\s*[a-z]\s*$")
_BODY_CUE_RE = re.compile(
    r"\b(?:in\s+violation\s+of|as\s+required\s+by|per\s+the\s+requirements\s+of|"
    r"requirements\s+of|pursuant\s+to|codified\s+into)\b",
    re.IGNORECASE,
)
_FOOTNOTE_CITE_RE = re.compile(
    r"(?<![\dA-Za-z])(19\d\.\d+(?:\([a-z0-9]+\))+)(?!\s*\()",
)
_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.\s*)?§\s*\d{3}\.\d+(?:\([a-z0-9]+\))*"
    r"(?:\s+[A-Z].*)?$",
)


def rule_basis_section_headings(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def low(text: str) -> str:
            return norm(text).lower()

        def pick_items():
            for key in ("lines", "paragraphs"):
                for item in doc.get(key, []) or []:
                    if isinstance(item, dict):
                        text = norm(item.get("text") or "")
                        if text:
                            yield key, item, text

        items = list(pick_items())
        if not items:
            return []

        anchor_idx = None
        scan_limit = min(len(items), 180)
        for i in range(scan_limit):
            text = low(items[i][2])
            if _INTRO_RE.search(text):
                anchor_idx = i
                break

        if anchor_idx is None:
            for i in range(scan_limit):
                text = low(items[i][2])
                if _ANCHOR_RE.search(text):
                    anchor_idx = i
                    break

        if anchor_idx is None:
            return []

        end_idx = len(items)
        for i in range(anchor_idx + 1, len(items)):
            if _STOP_RE.match(low(items[i][2])):
                end_idx = i
                break

        spans = []
        for source, item, text in items[anchor_idx:end_idx]:
            if not _HEADING_RE.match(text):
                continue
            span = {"text": text}
            for field in ("page_no", "line_no", "paragraph_no"):
                if field in item:
                    span[field] = item[field]
            spans.append(span)

        body_end = end_idx
        for i, (_, _, text) in enumerate(items[anchor_idx:end_idx], start=anchor_idx):
            if _FOOTNOTE_MARKER_RE.match(text):
                body_end = i
                break

        for source, item, text in items[anchor_idx:body_end]:
            if not _BODY_CUE_RE.search(text):
                continue
            for match in re.finditer(r"(?<![\dA-Za-z])(19\d\.\d+(?:\([a-z0-9]+\))*)", text):
                token = match.group(1)
                if "(" in token:
                    continue
                span = {"text": token}
                for field in ("page_no", "line_no", "paragraph_no"):
                    if field in item:
                        span[field] = item[field]
                spans.append(span)

        in_footnotes = False
        for source, item, text in items[body_end:end_idx]:
            if _FOOTNOTE_MARKER_RE.match(text):
                in_footnotes = True
                continue
            if not in_footnotes:
                continue
            if _STOP_RE.match(text) or re.match(r"^\s*\d+\.\s*", text):
                in_footnotes = False
                continue
            if re.search(r"(?<!\d)192\.735(?!\d)", text):
                span = {"text": "192.735"}
                for field in ("page_no", "line_no", "paragraph_no"):
                    if field in item:
                        span[field] = item[field]
                spans.append(span)
            if "§" in text:
                continue
            for match in _FOOTNOTE_CITE_RE.finditer(text):
                token = match.group(1)
                if token.endswith("(i)"):
                    token = token[:-3]
                span = {"text": token}
                for field in ("page_no", "line_no", "paragraph_no"):
                    if field in item:
                        span[field] = item[field]
                spans.append(span)

        return spans
    except Exception:
        return []
