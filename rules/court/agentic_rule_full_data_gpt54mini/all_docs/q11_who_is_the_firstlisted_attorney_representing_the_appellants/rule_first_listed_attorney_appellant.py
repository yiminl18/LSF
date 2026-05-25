import re


_COUNSEL_HEAD_RE = re.compile(
    r"(?:^|\n)\s*(?:COUNSEL|ATTORNEYS?)\s*(?:\n|$)",
    re.IGNORECASE,
)
_APPELLANT_TAG_RE = re.compile(
    r"(?:\bfor\b(?:[ \t]*\n[ \t]*|[ \t]+)?)?(?:"
    r"Plaintiffs?[\s\-]*Appellants?|"
    r"Defendants?[\s\-]*Appellants?|"
    r"Petitioners?[\s\-]*Appellants?|"
    r"Respondents?[\s\-]*Appellants?|"
    r"Appellants?"
    r")\b\.?",
    re.IGNORECASE,
)
_SIDE_TAG_RE = re.compile(
    r"(?:\bfor\b(?:[ \t]*\n[ \t]*|[ \t]+)?)?(?:"
    r"Plaintiffs?[\s\-]*Appellants?|"
    r"Plaintiffs?[\s\-]*Appellees?|"
    r"Defendants?[\s\-]*Appellants?|"
    r"Defendants?[\s\-]*Appellees?|"
    r"Petitioners?[\s\-]*Appellants?|"
    r"Petitioners?[\s\-]*Appellees?|"
    r"Respondents?[\s\-]*Appellants?|"
    r"Respondents?[\s\-]*Appellees?|"
    r"Appellants?|"
    r"Appellees?"
    r")\b\.?",
    re.IGNORECASE,
)


def rule_first_listed_attorney_appellant(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def first_attorney_from_block(block: str) -> str:
            text = norm(block)
            if not text:
                return ""

            text = re.sub(r"^\s*(?:and\s+)?", "", text, flags=re.IGNORECASE)
            text = text.split(";", 1)[0].strip()
            text = text.split(",", 1)[0].strip()
            text = re.split(r"\s+\band\b\s+", text, 1, flags=re.IGNORECASE)[0].strip()
            text = re.sub(r"\s*\((?:argued|lead|co[- ]?counsel|on brief|oral argument)\)\s*$", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
            text = re.sub(r"\s{2,}", " ", text).strip(" ,;:")
            if not text:
                return ""

            # If the extracted prefix still looks like an office or firm, keep it
            # only when there is no attorney-like name pattern to the left.
            return text

        def make_span(text: str, src: dict | None = None) -> dict:
            span = {"text": norm(text)}
            if src:
                for key in ("page_no", "paragraph_no", "line_no"):
                    if key in src:
                        span[key] = src[key]
            return span

        candidates: list[tuple[int, int, int, str, dict | None]] = []

        for source_key in ("pages", "paragraphs", "lines"):
            items = [item for item in (doc.get(source_key) or []) if isinstance(item, dict)]
            for idx, item in enumerate(items):
                raw_text = item.get("text") or ""
                text = norm(raw_text)
                if not text or "appellant" not in text.lower():
                    continue
                if not _COUNSEL_HEAD_RE.search(raw_text):
                    continue
                head = _COUNSEL_HEAD_RE.search(raw_text)
                search_start = head.end() if head else 0
                match = _APPELLANT_TAG_RE.search(raw_text, search_start)
                if match:
                    section = raw_text[search_start: match.start()]
                    if not section:
                        continue
                    prev = None
                    for side_match in _SIDE_TAG_RE.finditer(section):
                        prev = side_match
                    block = section[prev.end() :] if prev else section
                    block = block.strip()
                    if block:
                        candidates.append((0 if prev else 1, len(block), idx, block, item))

        raw_text = doc.get("text") or ""
        if isinstance(raw_text, str) and raw_text:
            head = _COUNSEL_HEAD_RE.search(raw_text)
            if head:
                search_start = head.end()
                match = _APPELLANT_TAG_RE.search(raw_text, search_start)
                if match:
                    section = raw_text[search_start: match.start()]
                    if section:
                        prev = None
                        for side_match in _SIDE_TAG_RE.finditer(section):
                            prev = side_match
                        block = section[prev.end() :] if prev else section
                        block = block.strip()
                        if block:
                            candidates.append((0 if prev else 1, len(block), 10000, block, None))

        if not candidates:
            for source_key in ("pages", "paragraphs", "lines"):
                items = [item for item in (doc.get(source_key) or []) if isinstance(item, dict)]
                combined = "\n".join((item.get("text") or "") for item in items if item.get("text"))
                if not combined:
                    continue
                head = _COUNSEL_HEAD_RE.search(combined)
                if not head:
                    continue
                search_start = head.end()
                match = _APPELLANT_TAG_RE.search(combined, search_start)
                if not match:
                    continue
                section = combined[search_start: match.start()]
                if not section:
                    continue
                prev = None
                for side_match in _SIDE_TAG_RE.finditer(section):
                    prev = side_match
                block = section[prev.end() :] if prev else section
                block = block.strip()
                if block:
                    candidates.append((0 if prev else 1, len(block), 20000, block, None))
                    break

        if not candidates:
            return []

        # Prefer the shortest prefix among plausible counsel blocks.
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _, _, _, prefix, src in candidates:
            first = first_attorney_from_block(prefix)
            if first:
                return [make_span(first, src)]

        return []
    except Exception:
        return []
