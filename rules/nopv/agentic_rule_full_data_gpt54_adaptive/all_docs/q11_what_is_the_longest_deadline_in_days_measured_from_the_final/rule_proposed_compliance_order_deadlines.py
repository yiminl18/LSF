import re


_HEADING_RE = re.compile(r"^\s*\ufeff*\f*PROPOSED\s+COMPLIANCE\s+ORDER\s*$", re.IGNORECASE)
_STOP_RE = re.compile(
    r"^(it\s+is\s+requested(?:\s*\(not\s+mandated\))?|response\s+options|response\s+to\s+this\s+notice|"
    r"enclosures?:|sincerely|respectfully|cc\b|attachments?)",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"^(page\s+\d+(?:\s+of\s+\d+)?)$", re.IGNORECASE)
_TOP_LEVEL_LABEL_RE = re.compile(r"^\s*(?:[A-Z]|\d+)\.\s*$")
_REPORT_ONLY_RE = re.compile(
    r"\b(report(?:s|ing)?|progress|follow[- ]?up|outstanding work necessary to implement|"
    r"days thereafter|until all work necessary to implement|91st day|every\s+\d+\s+days?)\b",
    re.IGNORECASE,
)
_MUST_RE = re.compile(r"\b(?:must|shall)\b", re.IGNORECASE)
_TIME_TOKEN = (
    r"(?:"
    r"\d{1,4}(?:\s*(?:1/2|½)|\s*\(\s*\d{1,4}\s*\))?"
    r"|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|eighteen|twenty|twenty-four)"
    r"(?:\s*\(\s*\d{1,4}\s*\))?"
    r")\s*,?\s*(?:calendar\s+)?(?:days?|months?|years?)"
)
_ANY_TIME_RE = re.compile(rf"\b(?P<phrase>{_TIME_TOKEN})\b", re.IGNORECASE)
_FINAL_ORDER_TIME_RE = re.compile(
    rf"(?P<phrase>{_TIME_TOKEN})\s+"
    rf"(?:of|from|after)\s+"
    rf"(?:(?:the\s+)?(?:receipt|issuance)\s+of\s+)?the\s+final\s+order\b",
    re.IGNORECASE,
)
_WORD_NUMS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "eighteen": 18,
    "twenty": 20,
    "twenty-four": 24,
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\ufeff", "").replace("\f", "")).strip()


def _materialize(items: list[dict], text_key: str) -> list[dict]:
    return [
        {
            "text": item.get(text_key) or "",
            "page_no": item.get("page_no"),
            "line_no": item.get("line_no"),
            "paragraph_no": item.get("paragraph_no"),
        }
        for item in items
        if isinstance(item, dict)
    ]


def _split_blocks(lines: list[dict], start_idx: int) -> list[list[dict]]:
    blocks: list[list[dict]] = []
    current: list[dict] = []
    for item in lines[start_idx + 1 :]:
        text = _norm(item.get("text") or "")
        if not text or _PAGE_RE.fullmatch(text):
            continue
        if _STOP_RE.match(text):
            break
        if _TOP_LEVEL_LABEL_RE.fullmatch(text):
            if current:
                blocks.append(current)
            current = [item]
            continue
        if current:
            current.append(item)
    if current:
        blocks.append(current)
    return blocks


def _phrase_to_days(phrase: str) -> int | None:
    text = re.sub(r"\s+", " ", (phrase or "").lower().replace(",", " ")).strip()
    unit_match = re.search(r"(?:calendar\s+)?(days?|months?|years?)$", text)
    if unit_match is None:
        return None
    unit = unit_match.group(1)
    value_text = text[: unit_match.start()].strip()

    paren_match = re.search(r"\(\s*(\d{1,4})\s*\)", value_text)
    if paren_match is not None:
        value = float(paren_match.group(1))
    else:
        frac_match = re.fullmatch(r"(\d{1,4})(?:\s*(1/2|½))?", value_text)
        if frac_match is not None:
            value = float(frac_match.group(1))
            if frac_match.group(2) is not None:
                value += 0.5
        else:
            word_text = re.sub(r"\(\s*\d{1,4}\s*\)", "", value_text).strip()
            if word_text not in _WORD_NUMS:
                return None
            value = float(_WORD_NUMS[word_text])

    if unit.startswith("day"):
        return int(round(value))
    if unit.startswith("month"):
        return int(round(value * 30))
    return int(round(value * 365))


def _sentence_spans(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _best_final_order_deadline_in_block(text: str) -> tuple[int, str] | None:
    best: tuple[int, str] | None = None
    for sentence in _sentence_spans(text):
        if _REPORT_ONLY_RE.search(sentence):
            continue
        for match in _FINAL_ORDER_TIME_RE.finditer(sentence):
            days = _phrase_to_days(match.group("phrase"))
            if days is None:
                continue
            if best is None or days > best[0]:
                best = (days, sentence)

    return best


def _best_fallback_deadline_in_block(text: str) -> tuple[int, str] | None:
    must_match = _MUST_RE.search(text)
    tail = text[must_match.start() :] if must_match is not None else text

    best: tuple[int, str] | None = None
    for sentence in _sentence_spans(tail):
        if _REPORT_ONLY_RE.search(sentence):
            continue
        for match in _ANY_TIME_RE.finditer(sentence):
            days = _phrase_to_days(match.group("phrase"))
            if days is None:
                continue
            if best is None or days > best[0]:
                best = (days, sentence)

    return best


def rule_proposed_compliance_order_deadlines(doc: dict) -> list[dict]:
    try:
        lines = _materialize(doc.get("lines") or [], "text")
        if not lines:
            lines = _materialize(doc.get("paragraphs") or [], "text")
        if not lines:
            raw_text = doc.get("text") or ""
            if not raw_text:
                return []
            lines = [{"text": raw_text, "page_no": None, "line_no": None, "paragraph_no": None}]

        heading_indices = [
            i for i, item in enumerate(lines) if _HEADING_RE.search(_norm(item.get("text") or ""))
        ]
        if not heading_indices:
            return []

        best_final_span: dict | None = None
        best_final_days = -1
        fallback_spans: list[tuple[int, dict]] = []

        for block in _split_blocks(lines, heading_indices[-1]):
            block_text = "\n".join(_norm(item.get("text") or "") for item in block if _norm(item.get("text") or ""))
            first = block[0]
            def make_span(text: str) -> dict:
                span = {"text": text}
                for key in ("page_no", "line_no", "paragraph_no"):
                    value = first.get(key)
                    if value is not None:
                        span[key] = value
                return span

            final_candidate = _best_final_order_deadline_in_block(block_text)
            if final_candidate is not None:
                days, best_text = final_candidate
                if days > best_final_days:
                    best_final_days = days
                    best_final_span = make_span(best_text)
                continue

            fallback_candidate = _best_fallback_deadline_in_block(block_text)
            if fallback_candidate is not None:
                days, best_text = fallback_candidate
                fallback_spans.append((days, make_span(best_text)))

        if best_final_span is not None:
            return [best_final_span]

        if fallback_spans:
            fallback_spans.sort(key=lambda item: item[0], reverse=True)
            seen: set[str] = set()
            selected: list[dict] = []
            for _, span in fallback_spans:
                text = span["text"]
                if text in seen:
                    continue
                seen.add(text)
                selected.append(span)
                if len(selected) == 3:
                    break
            return selected

        return []
    except Exception:
        return []
