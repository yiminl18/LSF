import re


_SEPARATE_OPINION_RE = re.compile(
    r"\b(?:dissent(?:ing)?|concurring(?: in part)?|concurrence|"
    r"separate opinion|opinion .* dissenting|statement .* dissenting)\b",
    re.IGNORECASE,
)
_DISPOSITION_RE = re.compile(
    r"\b(?:we\s+(?:therefore\s+|thus\s+|accordingly\s+|hereby\s+)?"
    r"(?:affirm|reverse|vacate|remand|dismiss|deny)\b"
    r"|(?:affirm|reverse|vacat(?:e|ed|ing)?|remand|dismiss|den(?:y|ied|ying)?|"
    r"affirmed|reversed|vacated|remanded|dismissed|denied)\b)",
    re.IGNORECASE,
)
_SPECIAL_PHRASES = (
    "opinion will issue in due course",
    "question certified; proceedings stayed",
    "the three-judge panel opinion is vacated",
    "this case is remanded",
    "this appeal is dismissed",
    "this appeal is denied",
    "the appeal is dismissed",
    "the appeal is denied",
)
_TAIL_CUE_RE = re.compile(
    r"(?:^\s*(?:AFFIRMED|REVERSED|VACATED|DISMISSED|DENIED|GRANTED|REMANDED)\b.*$|"
    r"\b(?:For that reason|Accordingly|Thus|Therefore|We therefore|We thus|We conclude|"
    r"We hold|We affirm|We reverse|We vacate|We dismiss|We deny|This appeal is|This case is)\b)",
    re.IGNORECASE | re.MULTILINE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\ufeff", "").replace("\f", "")).strip()


def _is_noisy(text: str) -> bool:
    s = _norm(text)
    if not s:
        return True
    low = s.lower()
    return bool(
        _SEPARATE_OPINION_RE.search(s)
        or "stay" in low
        or "certif" in low
    )


def _score_paragraph(text: str) -> int:
    s = _norm(text)
    if not s or _is_noisy(s):
        return -1

    low = s.lower()
    score = -1

    for phrase in _SPECIAL_PHRASES:
        if phrase in low:
            score = max(score, 120)

    if re.search(r"\bwe\s+(?:therefore\s+|thus\s+|accordingly\s+|hereby\s+)?"
                 r"(?:affirm|reverse|vacate|remand|dismiss|deny)\b", s, re.IGNORECASE):
        score = max(score, 110)

    if re.search(r"\b(?:affirm|reverse|vacat(?:e|ed|ing)?|remand|dismiss|den(?:y|ied|ying)?|"
                 r"grant|stay|certif(?:y|ied|ying))\b.*\b(?:district court|judgment|injunction|"
                 r"petition|motion|appeal|order|sentence|conviction|claim)\b", s, re.IGNORECASE):
        score = max(score, 95)

    if re.search(r"^\s*(?:accordingly|thus|therefore|in conclusion|we conclude|we hold)\b", s, re.IGNORECASE):
        score = max(score, 85)

    if re.search(r"\b(?:affirmed|reversed|vacated|dismissed|denied|granted|remanded)\b", s, re.IGNORECASE):
        score = max(score, 75)

    if s.isupper() and re.search(r"\b(?:AFFIRMED|REVERSED|VACATED|DISMISSED|DENIED|GRANTED|REMANDED|AFFIRM|REVERSE|VACATE|DISMISS|DENY|GRANT)\b", s):
        score = max(score, 100)

    return score


def _tail_snippet(text: str) -> str:
    s = (text or "").replace("\ufeff", "").strip()
    if not s:
        return ""

    sentence_re = re.compile(r"(?<=[.?!])\s+")
    parts = sentence_re.split(s.replace("\n", " "))

    best = ""
    best_score = -1
    best_idx = -1
    for idx, part in enumerate(parts):
        t = part.strip()
        if not t:
            continue
        low = t.lower()
        if "stay" in low or "certif" in low or _SEPARATE_OPINION_RE.search(t):
            continue

        score = -1
        if re.search(
            r"\b(?:for that reason|accordingly|thus|therefore)\s*,?\s*we\s+(?:affirm|reverse|vacate|remand|dismiss|deny)\b",
            t,
            re.IGNORECASE,
        ):
            score = max(score, 120)
        if re.search(
            r"\bwe\s+(?:therefore|thus|accordingly|hereby)\s+(?:affirm|reverse|vacate|remand|dismiss|deny)\b",
            t,
            re.IGNORECASE,
        ):
            score = max(score, 115)
        if re.search(r"\bwe\s+(?:affirm|reverse|vacate|remand|dismiss|deny)\b", t, re.IGNORECASE):
            score = max(score, 105)
        if re.search(r"^\s*(?:AFFIRMED|REVERSED|VACATED|DISMISSED|DENIED|GRANTED|REMANDED)\b", t):
            score = max(score, 100)
        if re.search(r"\b(?:affirmed|reversed|vacated|dismissed|denied|granted|remanded)\b", t, re.IGNORECASE):
            score = max(score, 90)
        if re.search(r"\b(?:We hold|We conclude)\b", t, re.IGNORECASE) and re.search(
            r"\b(?:affirm|reverse|vacate|remand|dismiss|deny)\b", t, re.IGNORECASE
        ):
            score = max(score, 95)

        if score > best_score or (score == best_score and idx > 0):
            best = t
            best_score = score
            best_idx = idx

    if best_idx >= 0 and best_idx + 1 < len(parts):
        next_part = parts[best_idx + 1].strip()
        next_low = next_part.lower()
        if next_part and not _SEPARATE_OPINION_RE.search(next_part):
            if re.search(r"\bremand\b", next_low) or re.match(
                r"^\s*(?:AFFIRMED|REVERSED|VACATED|DISMISSED|DENIED|GRANTED|REMANDED)\b",
                next_part,
                re.IGNORECASE,
            ):
                if best and not best.endswith(next_part):
                    best = f"{best} {next_part}".strip()

    return best if best_score >= 0 else s


def rule_panel_disposition_conclusion(doc: dict) -> list[dict]:
    try:
        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        if not paragraphs:
            raw = _norm(doc.get("text") or "")
            if not raw:
                return []
            paragraphs = [{"text": raw}]

        best = None
        best_score = -1
        best_index = -1

        for idx, para in enumerate(paragraphs):
            text = para.get("text") or ""
            score = _score_paragraph(text)
            if score < 0:
                continue
            para_index = para.get("paragraph_index")
            order = para_index if isinstance(para_index, int) else idx
            if score > best_score or (score == best_score and order > best_index):
                best = para
                best_score = score
                best_index = order

        if not best or best_score < 0:
            return []

        snippet = _tail_snippet(best.get("text") or "")
        if not snippet:
            return []

        span = {"text": snippet}
        for key in ("page_no", "paragraph_no", "paragraph_index"):
            value = best.get(key)
            if value is not None:
                span[key] = value
        return [span]
    except Exception:
        return []
