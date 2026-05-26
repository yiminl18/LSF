import re


UNEMP_RE = re.compile(r"un\s*em\s*ploy\s*ment\s*rate|civilian\s*unemployment\s*rate", re.IGNORECASE)
PERCENT_RE = re.compile(r"\b\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)\b", re.IGNORECASE)
MONTH_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
DATE_MARKERS_RE = re.compile(
    r"\b(?:recent|latest|current|now|as of|in|during|since|stood at|fell to|rose to|edged down to|dropped to|declined to|hovered|reached|hit|was|were)\b",
    re.IGNORECASE,
)
RATE_WORDS_RE = re.compile(r"\b(?:percent|per\s*cent)\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    text = text or ""
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize(text).lower())


def _score_text(text: str) -> int:
    low = text.lower()
    compact = _compact(text)
    score = 0
    if "unemploymentrate" in compact or "civilianunemploymentrate" in compact:
        score += 6
    if PERCENT_RE.search(low):
        score += 4
    if MONTH_YEAR_RE.search(low):
        score += 3
    elif MONTH_RE.search(low):
        score += 2
    if DATE_MARKERS_RE.search(low):
        score += 2
    if any(tok in compact for tok in ("edgeddown", "fellt", "roset", "stoodat", "hovered", "droppedto", "declinedto", "reached", "hit", "stable")):
        score += 1
    return score


def _sentence_spans(text: str) -> list[str]:
    text = _normalize(text)
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", text)
    if len(chunks) == 1:
        return [text]
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _extract_rate_phrase(text: str) -> str | None:
    text = _normalize(text)
    if not text:
        return None
    patterns = [
        r"(?:stood at|stood|was|were)\s+(?P<rate>\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent))",
        r"(?:edged down to|fell to|dropped to|declined to|rose to|increased to|has risen to|has fallen to|has crept lower to)\s+(?P<rate>\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent))",
        r"(?:reached|peaked at|hit)\s+(?:a\s+[^,.;:]+?\s+)?(?:of\s+)?(?P<rate>\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent))",
        r"(?:from|between)\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)\s+(?:to|and)\s+(?P<rate>\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent))",
        r"range of\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)\s+(?:to|and)\s+(?P<rate>\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent))",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            rate = _normalize(m.group("rate"))
            if rate:
                return rate
    matches = list(PERCENT_RE.finditer(text))
    if not matches:
        return None
    # Prefer the last percentage when the sentence describes a range or a before/after comparison.
    return _normalize(matches[-1].group(0))


def _extract_context_phrase(text: str) -> str | None:
    text = _normalize(text)
    if not text:
        return None
    patterns = [
        r"rose from\s+.*?\s+to\s+.*?\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)(?:\s+(?:in|as of)\s+[A-Za-z]+\s+\d{4})?",
        r"(?:stood at|stood|was|were)\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)",
        r"(?:edged down to|fell to|dropped to|declined to|rose to|increased to|has risen to|has fallen to|has crept lower to)\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)",
        r"(?:reached|peaked at|hit)\s+(?:a\s+[^,.;:]+?\s+)?(?:of\s+)?\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)(?:\s+(?:in|as of)\s+[A-Za-z]+\s+\d{4})?",
        r"(?:from|between)\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)\s+(?:to|and)\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)",
        r"range of\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)\s+(?:to|and)\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)",
        r"hovered in a narrow range of\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)\s+(?:to|and)\s+\d{1,2}(?:\.\d+)?\s*(?:per\s*cent|percent)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return _normalize(m.group(0))
    return None


def rule_unemployment_rate_recent(doc: dict) -> list[dict]:
    try:
        best_span = None
        best_score = 0
        seen = set()

        paragraphs = doc.get("paragraphs") or []
        for para in paragraphs:
            text = para.get("text") or ""
            if not text:
                continue
            compact = _compact(text)
            if "unemploymentrate" not in compact and "civilianunemploymentrate" not in compact:
                continue
            sentences = _sentence_spans(text)
            best = None
            best_score = 0
            for sent in sentences:
                score = _score_text(sent)
                if score > best_score:
                    best = sent
                    best_score = score
            if best is None:
                best = _normalize(text)
                best_score = _score_text(best)
            if best_score < 8:
                continue
            phrase = _extract_rate_phrase(best)
            context = _extract_context_phrase(best)
            if phrase and context and phrase != context:
                best_context = context
                best = phrase
            else:
                best_context = context
                if phrase:
                    best = phrase
            key = (para.get("page_no"), para.get("paragraph_no"), best)
            if key in seen:
                continue
            seen.add(key)
            if best_score > 0 and (best_span is None or best_score > best_span[0]):
                spans_to_return = []
                if best_context and best_context != best:
                    span2 = {"text": best_context}
                    if para.get("page_no") is not None:
                        span2["page_no"] = para["page_no"]
                    if para.get("paragraph_no") is not None:
                        span2["paragraph_no"] = para["paragraph_no"]
                    spans_to_return.append(span2)
                span = {"text": best}
                spans_to_return.append(span)
                best_span = (best_score, spans_to_return)

        lines = doc.get("lines") or []
        for idx, line in enumerate(lines):
            text = line.get("text") or ""
            if not text:
                continue
            compact = _compact(text)
            if "unemploymentrate" not in compact and "civilianunemploymentrate" not in compact:
                continue
            window_parts = []
            for j in range(max(0, idx - 2), min(len(lines), idx + 4)):
                part = _normalize(lines[j].get("text") or "")
                if part:
                    window_parts.append(part)
            window = " ".join(window_parts).strip()
            if not window:
                continue
            score = _score_text(window)
            if score < 8:
                continue
            phrase = _extract_rate_phrase(window)
            context = _extract_context_phrase(window)
            if phrase:
                window = phrase
            key = (line.get("page_no"), line.get("line_no"), window)
            if key in seen:
                continue
            seen.add(key)
            if best_span is None or score > best_span[0]:
                spans_to_return = []
                if context and context != window:
                    span2 = {"text": context}
                    if line.get("page_no") is not None:
                        span2["page_no"] = line["page_no"]
                    if line.get("line_no") is not None:
                        span2["line_no"] = line["line_no"]
                    spans_to_return.append(span2)
                span = {"text": window}
                spans_to_return.append(span)
                best_span = (score, spans_to_return)

        if best_span is None:
            return []
        return best_span[1]
    except Exception:
        return []
