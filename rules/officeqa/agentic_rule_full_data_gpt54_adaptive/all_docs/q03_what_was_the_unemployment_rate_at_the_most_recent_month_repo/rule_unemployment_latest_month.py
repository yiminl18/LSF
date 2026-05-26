import re


MONTH_ALIASES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

MONTH_TOKEN_RE = re.compile(
    r"\b("
    r"January|Jan\.?|February|Feb\.?|March|Mar\.?|April|Apr\.?|May|June|Jun\.?|"
    r"July|Jul\.?|August|Aug\.?|September|Sept?\.?|October|Oct\.?|November|Nov\.?|"
    r"December|Dec\.?"
    r")\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    r"\b("
    r"January|Jan\.?|February|Feb\.?|March|Mar\.?|April|Apr\.?|May|June|Jun\.?|"
    r"July|Jul\.?|August|Aug\.?|September|Sept?\.?|October|Oct\.?|November|Nov\.?|"
    r"December|Dec\.?"
    r")\s+(\d{4})\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"\b\d{1,2}(?:\.\d+)?\s*(?:%|percent|per\s*cent)\b", re.IGNORECASE)
UNEMP_TOKEN_RE = re.compile(r"\bunemployment\b", re.IGNORECASE)
VERB_RE = re.compile(
    r"\b("
    r"stood at|was|were|fell to|fell below|declined to|declining to|edged down to|"
    r"edged up to|ticked up to|ticked down to|dipped to|rose to|jumped to|climbed to|"
    r"receded to|retreated to|eased to|eased by|has declined|has fallen|has eased|"
    r"has dropped|has risen|hovered|remained at|held at"
    r")\b",
    re.IGNORECASE,
)
RANGE_RE = re.compile(
    r"\b(?:between|from|range of|hovered between|hovered in a narrow range of)\b[^.]{0,100}?"
    r"\d{1,2}(?:\.\d+)?\s*(?:%|percent|per\s*cent)\s*(?:to|and)\s*"
    r"\d{1,2}(?:\.\d+)?\s*(?:%|percent|per\s*cent)\b",
    re.IGNORECASE,
)
DIRECT_RATE_RE = re.compile(
    r"\b(?:headline\s+|civilian\s+)?(?:unemployment rate|jobless rate)\b[^.]{0,220}?"
    r"(?:stood at|was|were|fell to|fell below|declined to|declining to|edged down to|"
    r"edged up to|ticked up to|ticked down to|dipped to|rose to|jumped to|climbed to|"
    r"receded to|retreated to|eased to|has declined[^.]{0,20}?to|has fallen[^.]{0,20}?to|"
    r"had fallen[^.]{0,20}?to|had declined[^.]{0,20}?to|falling to|remained at|held at|"
    r"stabilized at|stabilised at)\s+\d{1,2}(?:\.\d+)?\s*(?:%|percent|per\s*cent)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    text = text or ""
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"(\w)\s*-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize(text).lower())


def _split_sentences(text: str) -> list[str]:
    norm = _normalize(text)
    if not norm:
        return []
    parts = re.split(r"(?<=[.!?])\s+", norm)
    return [part.strip() for part in parts if part.strip()]


def _has_primary_rate(text: str) -> bool:
    comp = _compact(text)
    return any(
        token in comp
        for token in (
            "unemploymentrate",
            "civilianunemploymentrate",
            "headlineunemploymentrate",
            "joblessrate",
        )
    )


def _has_bad_context(text: str) -> bool:
    comp = _compact(text)
    return any(
        token in comp
        for token in (
            "unemploymenttrustfund",
            "unemploymentinsurance",
            "emergencyunemploymentcompensation",
            "unemploymentassistance",
            "federalunemploymenttaxact",
        )
    )


def _u6_penalty(text: str) -> int:
    comp = _compact(text)
    penalty = 0
    for token in (
        "u6unemploymentrate",
        "broadermeasureofunemployment",
        "underemployed",
        "marginallyattached",
        "27weeks",
        "longtermunemployment",
        "shareofthelaborforcewhowereunemployed",
    ):
        if token in comp:
            penalty += 40
    return penalty


def _extract_latest_date_score(text: str, doc_year: int | None) -> int:
    latest = 0
    for month, year in MONTH_YEAR_RE.findall(text):
        month_key = month.lower().rstrip(".")
        month_no = MONTH_ALIASES.get(month_key)
        if month_no:
            year_num = int(year)
            bonus = 24 if doc_year is not None and year_num == doc_year else 12
            latest = max(latest, bonus + month_no)
    if latest:
        return latest
    if doc_year is None:
        return 0
    for month in MONTH_TOKEN_RE.findall(text):
        month_key = month.lower().rstrip(".")
        month_no = MONTH_ALIASES.get(month_key)
        if month_no:
            latest = max(latest, 12 + month_no)
    return latest


def _count_percents(text: str) -> int:
    return len(PERCENT_RE.findall(text))


def _is_range_only(text: str) -> bool:
    norm = _normalize(text)
    if not RANGE_RE.search(norm):
        return False
    return not (
        "stood at" in norm.lower()
        or "ticked up to" in norm.lower()
        or "edged up to" in norm.lower()
        or "edged down to" in norm.lower()
        or "fell to" in norm.lower()
        or "declined to" in norm.lower()
        or "rose to" in norm.lower()
        or "jumped to" in norm.lower()
        or "climbed to" in norm.lower()
        or "receded to" in norm.lower()
        or "retreated to" in norm.lower()
        or "eased to" in norm.lower()
        or "falling to" in norm.lower()
    )


def _score_candidate(text: str, doc_year: int | None, paragraph_index: int) -> int:
    norm = _normalize(text)
    low = norm.lower()
    comp = _compact(norm)
    has_direct_rate = bool(DIRECT_RATE_RE.search(norm))
    if not norm or _has_bad_context(norm):
        return -1
    if not _has_primary_rate(norm):
        return -1
    if not PERCENT_RE.search(norm):
        return -1
    if _is_range_only(norm):
        return -1
    if "table of contents" in low:
        return -1
    if len(norm) > 450:
        return -1
    if (
        any(token in low for token in ("fomc", "federal funds rate", "committee's", "committee indicated"))
        and not has_direct_rate
    ):
        return -1
    if not has_direct_rate and "unemployment rate" in low and "(percent)" in low:
        return -1
    if "unemployment rate (" in low and _count_percents(norm) <= 1 and not VERB_RE.search(norm):
        return -1

    score = 0
    score += _extract_latest_date_score(norm, doc_year) * 8
    if "unemploymentrate" in comp or "civilianunemploymentrate" in comp:
        score += 60
    if "headlineunemploymentrate" in comp or "joblessrate" in comp:
        score += 30
    if has_direct_rate:
        score += 40
    if VERB_RE.search(norm):
        score += 15
    if norm.lower().startswith("the unemployment rate"):
        score += 10
    if MONTH_TOKEN_RE.search(norm):
        score += 10
    if _count_percents(norm) == 1:
        score += 12
    elif _count_percents(norm) == 2:
        score += 4
    else:
        score -= 8 * (_count_percents(norm) - 2)
    if not has_direct_rate and len(norm) > 220:
        score -= 45
    score -= max(0, len(norm) - 180) // 12
    if "peak" in low or "high of" in low or "low of" in low:
        score -= 6
    if "average" in low or "averaged" in low:
        score -= 8
    score -= _u6_penalty(norm)
    score -= paragraph_index
    return score


def rule_unemployment_latest_month(doc: dict) -> list[dict]:
    try:
        doc_name = (doc.get("doc_name") or "").lower()
        year_match = re.search(r"treasury_bulletin_(\d{4})_", doc_name)
        doc_year = int(year_match.group(1)) if year_match else None

        best_text = None
        best_meta = None
        best_score = -1

        for para_idx, para in enumerate(doc.get("paragraphs") or []):
            raw = para.get("text") or ""
            norm = _normalize(raw)
            if not norm:
                continue
            comp = _compact(norm)
            if "unemployment" not in comp and "jobless" not in comp:
                continue

            sentences = _split_sentences(raw)
            if not sentences:
                sentences = [norm]

            candidates = []
            for i, sentence in enumerate(sentences):
                if sentence:
                    candidates.append(sentence)
                if i > 0:
                    candidates.append(f"{sentences[i - 1]} {sentence}")
                if i + 1 < len(sentences):
                    candidates.append(f"{sentence} {sentences[i + 1]}")
            candidates.append(norm)

            seen = set()
            for candidate in candidates:
                candidate_norm = _normalize(candidate)
                if not candidate_norm:
                    continue
                key = candidate_norm.lower()
                if key in seen:
                    continue
                seen.add(key)
                score = _score_candidate(candidate_norm, doc_year, para_idx)
                if score <= best_score:
                    continue
                best_score = score
                best_text = candidate_norm
                best_meta = {
                    "page_no": para.get("page_no"),
                    "paragraph_no": para.get("paragraph_no"),
                }

        if best_text is None:
            return []

        span = {"text": best_text}
        if best_meta:
            if best_meta.get("page_no") is not None:
                span["page_no"] = best_meta["page_no"]
            if best_meta.get("paragraph_no") is not None:
                span["paragraph_no"] = best_meta["paragraph_no"]
        return [span]
    except Exception:
        return []
