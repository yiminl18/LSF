import re


_PART_RE = r"(?:40|1\d\d|2\d\d)\.\d+(?:\([A-Za-z0-9]+\))*"
_HEADING_RE = re.compile(
    rf"^\s*(?:\d+\.\s*)?(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?§\s*({_PART_RE})\b",
    re.IGNORECASE,
)
_ITEM_RE = re.compile(r"^\s*\d+\.\s*$")
_STOP_RE = re.compile(
    r"^(proposed civil penalty|warning items|proposed compliance order|response to this notice)\b",
    re.IGNORECASE,
)
_ALLEGATION_RE = re.compile(
    r"\b(failed|did not|didn't|was not|were not|has not|have not)\b",
    re.IGNORECASE,
)
_NOISE_RE = re.compile(r"^\s*(?:cpf\b|page\b|\d+\s*$)", re.IGNORECASE)
_SUBSECTION_RE = re.compile(r"^\s*\([A-Za-z0-9ivxlcdm]+\)", re.IGNORECASE)


def _trim_lines(lines):
    start_idx = None
    for idx in range(len(lines)):
        window = " ".join(" ".join(lines[j]["text"].lower().split()) for j in range(idx, min(len(lines), idx + 3)))
        if ("item inspected" in window or "items inspected" in window) and "probable violation" in window:
            start_idx = idx + 1
            break
    if start_idx is None:
        return []

    trimmed = []
    for line in lines[start_idx:]:
        text = line["text"].strip()
        if _STOP_RE.match(text):
            break
        trimmed.append(line)
    return trimmed


def _append(spans, seen, line):
    key = (line["page_no"], line["line_no"], line["text"])
    if key in seen:
        return
    seen.add(key)
    spans.append({"text": line["text"], "page_no": line["page_no"], "line_no": line["line_no"]})


def rule_violation_blocks(doc: dict) -> list[dict]:
    try:
        lines = _trim_lines(doc.get("lines", []))
        spans = []
        seen = set()
        context = 0
        after_item = False

        for line in lines:
            text = line["text"].strip()
            if not text:
                if context > 0:
                    context -= 1
                continue

            is_noise = _NOISE_RE.match(text) or ".docx" in text.lower()

            if _ITEM_RE.match(text):
                after_item = True
                context = 0
                continue

            if _HEADING_RE.match(text):
                heading = _HEADING_RE.match(text)
                if not is_noise:
                    _append(spans, seen, line)
                after_item = False
                base = heading.group(1).split("(", 1)[0] if heading else ""
                if base == "195.446":
                    context = 12
                elif base == "192.1007":
                    context = 0
                else:
                    context = 6
                continue

            if after_item and _HEADING_RE.match(text):
                heading = _HEADING_RE.match(text)
                if not is_noise:
                    _append(spans, seen, line)
                after_item = False
                base = heading.group(1).split("(", 1)[0] if heading else ""
                if base == "195.446":
                    context = 12
                elif base == "192.1007":
                    context = 0
                else:
                    context = 6
                continue

            if _ALLEGATION_RE.search(text):
                if not is_noise:
                    _append(spans, seen, line)
                context = 0
                after_item = False
                continue

            if context > 0 and not is_noise:
                if _SUBSECTION_RE.match(text) or text[0].isalnum():
                    _append(spans, seen, line)
                    context -= 1
                    continue

            after_item = False

        return spans
    except Exception:
        return []
