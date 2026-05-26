import re


_PART_RE = r"(?:40|1\d\d|2\d\d)\.\d+(?:\([A-Za-z0-9]+\))*"
_CITE_RE = re.compile(
    rf"((?:49\s*C\.?\s*F\.?\s*R\.?\s*)?§\s*{_PART_RE}|section[s]?\s+{_PART_RE})",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"^(proposed civil penalty|warning items|proposed compliance order|response to this notice)\b",
    re.IGNORECASE,
)
_ALLEGATION_RE = re.compile(
    r"\b(failed|did not|didn't|was not|were not|has not|have not)\b",
    re.IGNORECASE,
)
_EARLY_ALLEGATION_RE = re.compile(
    r"^\s*(?:therefore,?\s+|thus,?\s+)?[^§\n]{0,100}\b(failed|did not|didn't|was not|were not|has not|have not)\b",
    re.IGNORECASE,
)
_CONCLUSION_RE = re.compile(r"^\s*(therefore|thus)\b", re.IGNORECASE)
_SECTION_START_RE = re.compile(r"(item inspected|items inspected)", re.IGNORECASE)
_PART_RE = r"(?:40|1\d\d|2\d\d)\.\d+(?:\([A-Za-z0-9]+\))*"
_LEAD_CITE_RE = re.compile(rf"(?:§+|section[s]?)\s*({_PART_RE})", re.IGNORECASE)
_FOLLOW_CITE_RE = re.compile(rf"\b(?:and|or|,)\s*({_PART_RE})", re.IGNORECASE)


def _line_window(lines):
    start_idx = None
    for idx in range(len(lines)):
        window = " ".join(" ".join(lines[j]["text"].lower().split()) for j in range(idx, min(len(lines), idx + 3)))
        if _SECTION_START_RE.search(window) and "probable violation" in window:
            start_idx = idx + 1
            break

    if start_idx is None:
        return

    for line in lines[start_idx:]:
        text = line["text"].strip()
        if _STOP_RE.match(text):
            break
        yield line


def rule_citation_lines(doc: dict) -> list[dict]:
    try:
        spans = []
        seen = set()
        for line in _line_window(doc.get("lines", [])):
            text = line["text"].strip()
            if not text:
                continue
            if not (_EARLY_ALLEGATION_RE.search(text) or _CONCLUSION_RE.match(text)):
                continue
            if not _CITE_RE.search(text):
                continue

            key = (line["page_no"], line["line_no"], line["text"])
            if key not in seen:
                seen.add(key)
                spans.append(
                    {
                        "text": line["text"],
                        "page_no": line["page_no"],
                        "line_no": line["line_no"],
                    }
                )

            citations = _LEAD_CITE_RE.findall(text)
            if "§§" in text or "sections " in text.lower():
                citations.extend(_FOLLOW_CITE_RE.findall(text))

            deduped = []
            for citation in citations:
                if citation not in deduped:
                    deduped.append(citation)

            for citation in deduped:
                nested_levels = citation.count("(")
                if len(deduped) > 1 or nested_levels >= 2:
                    synth_key = ("synthetic", citation)
                    if synth_key in seen:
                        continue
                    seen.add(synth_key)
                    spans.append({"text": f"49 CFR § {citation}"})
        return spans
    except Exception:
        return []
