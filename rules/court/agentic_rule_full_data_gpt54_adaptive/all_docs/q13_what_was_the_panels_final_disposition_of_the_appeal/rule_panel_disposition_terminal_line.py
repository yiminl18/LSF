import re


_SEPARATE_OPINION_RE = re.compile(
    r"\b(?:dissent(?:ing)?|concurring(?: in part)?|concurrence|"
    r"separate opinion|opinion .* dissenting|statement .* dissenting)\b",
    re.IGNORECASE,
)
_LINE_DISPOSITION_RE = re.compile(
    r"^(?:"
    r"affirmed(?: in part)?(?:[,.;].*)?|reversed(?: in part)?(?:[,.;].*)?|"
    r"vacated(?: in part)?(?:[,.;].*)?|dismissed(?:[,.;].*)?|denied(?:[,.;].*)?|"
    r"granted(?:[,.;].*)?|remanded(?:[,.;].*)?|affirm(?:[,.;].*)?|reverse(?:[,.;].*)?|"
    r"vacate(?:[,.;].*)?|dismiss(?:[,.;].*)?|deny(?:[,.;].*)?|grant(?:[,.;].*)?|"
    r"question certified; proceedings stayed\.?|administrative stay granted\.?|"
    r"an opinion will issue in due course\.?|"
    r"(?:oral argument .*? the court will file a new opinion in due course\.?)|"
    r"the three-judge panel opinion is vacated\.)$",
    re.IGNORECASE,
)
_PARA_DISPOSITION_RE = re.compile(
    r"\bwe\s+(?:therefore\s+|thus\s+|accordingly\s+|hereby\s+)?"
    r"(?:affirm|reverse|vacate|remand|dismiss|deny)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\ufeff", "").replace("\f", "")).strip()


def _is_noisy(text: str) -> bool:
    s = _norm(text)
    if not s:
        return True
    return bool(_SEPARATE_OPINION_RE.search(s))


def _score_line(text: str) -> int:
    s = _norm(text)
    if not s or _is_noisy(s):
        return -1

    low = s.lower()
    if _LINE_DISPOSITION_RE.match(s):
        return 120

    if any(
        phrase in low
        for phrase in (
            "opinion will issue in due course",
            "court will file a new opinion in due course",
            "administrative stay granted",
            "question certified; proceedings stayed",
            "the three-judge panel opinion is vacated",
        )
    ):
        return 110

    if len(s) <= 180 and re.search(r"\b(?:affirmed|reversed|vacated|dismissed|denied|granted|remanded)\b", s, re.IGNORECASE):
        return 90

    if len(s) <= 120 and re.search(r"\b(?:affirm|reverse|vacate|dismiss|deny|grant)\b", s, re.IGNORECASE):
        return 70

    return -1


def _has_paragraph_candidate(paragraphs: list[dict]) -> bool:
    for para in paragraphs:
        s = _norm(para.get("text") or "")
        if not s:
            continue
        low = s.lower()
        if "stay" in low or "certif" in low or _SEPARATE_OPINION_RE.search(s):
            if "opinion will issue in due course" not in low and "court will file a new opinion in due course" not in low:
                continue
        if any(
            phrase in low
            for phrase in (
                "opinion will issue in due course",
                "court will file a new opinion in due course",
            )
        ):
            return True
        if _PARA_DISPOSITION_RE.search(s):
            return True
        if re.search(
            r"\b(?:affirm|reverse|vacat(?:e|ed|ing)?|remand|dismiss|den(?:y|ied|ying)?)\b.*"
            r"\b(?:district court|judgment|injunction|petition|motion|appeal|order|sentence|conviction|claim)\b",
            s,
            re.IGNORECASE,
        ):
            return True
    return False


def _trim_line(text: str) -> str:
    s = (text or "").replace("\ufeff", "").strip()
    if not s:
        return ""

    parts = re.split(r"(?<=[.?!])\s+", s.replace("\n", " "))
    for part in parts:
        t = part.strip()
        if not t:
            continue
        low = t.lower()
        if "opinion will issue in due course" in low:
            return t
        if "court will file a new opinion in due course" in low:
            return t
        if _LINE_DISPOSITION_RE.match(t):
            return t
        if re.search(r"^\s*(?:AFFIRMED|REVERSED|VACATED|DISMISSED|DENIED|GRANTED|REMANDED)\b", t, re.IGNORECASE):
            return t

    return s


def _extract_due_course_sentence(text: str) -> str:
    s = _norm(text)
    if not s:
        return ""

    for pattern in (
        r"[^.?!]*an opinion will issue in due course\.",
        r"[^.?!]*the court will file a new opinion in due course\.",
    ):
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return s


def rule_panel_disposition_terminal_line(doc: dict) -> list[dict]:
    try:
        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        if _has_paragraph_candidate(paragraphs):
            return []

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            raw = _norm(doc.get("text") or "")
            if not raw:
                return []
            lines = [{"text": raw}]

        best = None
        best_score = -1
        best_index = -1

        for idx, line in enumerate(lines):
            text = line.get("text") or ""
            score = _score_line(text)
            if score < 0:
                continue
            line_index = line.get("line_index")
            order = line_index if isinstance(line_index, int) else idx
            if score > best_score or (score == best_score and order > best_index):
                best = line
                best_score = score
                best_index = order

        if not best or best_score < 0:
            return []

        snippet = _trim_line(best.get("text") or "")
        if not snippet:
            return []

        if "opinion will issue in due course" in snippet.lower() or "court will file a new opinion in due course" in snippet.lower():
            paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
            for para in paragraphs:
                para_text = _norm(para.get("text") or "")
                if (
                    "opinion will issue in due course" in para_text.lower()
                    or "court will file a new opinion in due course" in para_text.lower()
                ):
                    snippet = _extract_due_course_sentence(para.get("text") or snippet)
                    break

        span = {"text": snippet}
        for key in ("page_no", "line_no", "line_index"):
            value = best.get(key)
            if value is not None:
                span[key] = value
        return [span]
    except Exception:
        return []
