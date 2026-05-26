import re


_COUNSEL_HEADING_RE = re.compile(r"^\s*COUNSEL\s*$", re.IGNORECASE)
_APPELLEE_BLOCK_RE = re.compile(
    r"\bfor\s+"
    r"(?:Plaintiff|Plaintiffs|Defendant|Defendants|Respondent|Respondents|"
    r"Intervenor(?:-Defendant|-Respondent)?|Appellee|Appellees)"
    r"[^.\n]*\bAppellee(?:s)?\b",
    re.IGNORECASE,
)
_COUNSEL_LIST_HEADING_RE = re.compile(
    r"^\s*Counsel for\s+.*Appellee(?:s)?\s*:?\s*(.*)$", re.IGNORECASE
)
_MAJOR_STOP_HEADING_RE = re.compile(
    r"^\s*(?:OPINION|ORDER|SUMMARY|JUDGMENT|MEMORANDUM|BACKGROUND|INTRODUCTION|"
    r"DISCUSSION|CONCLUSION|PER CURIAM)\b",
    re.IGNORECASE,
)
_ORGANIZATION_WORD_RE = re.compile(
    r"\b(?:LLP|LLC|P\.C\.|PC|INC\.?|CORPORATION|COMPANY|DEPARTMENT|OFFICE|"
    r"UNIVERSITY|ASSOCIATION|GROUP|SERVICES|ATTORNEY|ATTORNEYS|COUNSEL|"
    r"LAWYERS|PARTNERS|BUDD|CLINIC|FOUNDATION|TRUST|COMMITTEE|AGENCY|"
    r"STATES ATTORNEY|UNITED STATES ATTORNEY)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def _clean_name(text: str) -> str | None:
    text = _norm(text)
    if not text:
        return None
    # Remove a leading counsel-list label if one is present.
    text = re.sub(r"^\s*Counsel for\s+.*?Appellee(?:s)?\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        return None
    # The first attorney is typically the first comma-delimited token.
    candidate = text.split(",", 1)[0].strip()
    if " and " in candidate:
        candidate = candidate.split(" and ", 1)[0].strip()
    candidate = re.sub(r"\s*\([^)]*\)\s*$", "", candidate).strip(" ,;")
    candidate = candidate.strip()
    if not candidate or len(candidate) < 2:
        return None
    if ":" in candidate or "@" in candidate or _ORGANIZATION_WORD_RE.search(candidate):
        return None
    if len(candidate.split()) < 2:
        return None
    return candidate


def _line_dicts(doc: dict) -> list[dict]:
    lines = doc.get("lines") or []
    if lines:
        return lines
    text = doc.get("text") or ""
    out = []
    for i, raw in enumerate(text.splitlines(), start=1):
        out.append({"page_no": None, "line_no": i, "text": raw})
    return out


def _span_from_lines(lines: list[dict], start: int, end: int, text: str) -> dict:
    span = {"text": text}
    if 0 <= start < len(lines):
        page_no = lines[start].get("page_no")
        line_no = lines[start].get("line_no")
        if page_no is not None:
            span["page_no"] = page_no
        if line_no is not None:
            span["line_no"] = line_no
    return span


def _is_party_end(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    party_tokens = (
        "appellant",
        "appellants",
        "petitioner",
        "petitioners",
        "respondent",
        "respondents",
        "appellee",
        "appellees",
        "amicus",
        "amici",
        "intervenor",
        "plaintiff",
        "defendant",
    )
    if "for" in compact and any(token in compact for token in party_tokens):
        return True
    if len(compact) <= 60 and compact.endswith(".") and not any(ch in compact for ch in ",;:@") and any(
        token in compact for token in party_tokens
    ):
        return True
    return False


def _extract_from_standard_counsel(lines: list[dict]) -> list[dict]:
    counsel_idx = None
    for i, line in enumerate(lines):
        if _COUNSEL_HEADING_RE.match(_norm(line.get("text", ""))):
            counsel_idx = i
            break
    if counsel_idx is None:
        return []

    for end in range(counsel_idx + 1, len(lines)):
        raw = _norm(lines[end].get("text", ""))
        if not raw:
            continue
        if _MAJOR_STOP_HEADING_RE.match(raw):
            break

        window_start = max(counsel_idx + 1, end - 2)
        window = " ".join(
            _norm(lines[j].get("text", "")) for j in range(window_start, end + 1) if _norm(lines[j].get("text", ""))
        )
        compact = re.sub(r"\s+", "", window).lower()
        if "for" not in compact or "appellee" not in compact:
            continue

        start = end
        while start > counsel_idx + 1:
            prev = _norm(lines[start - 1].get("text", ""))
            if not prev:
                start -= 1
                continue
            if _MAJOR_STOP_HEADING_RE.match(prev):
                break
            if _is_party_end(prev):
                break
            start -= 1

        block = " ".join(
            _norm(lines[j].get("text", "")) for j in range(start, end + 1) if _norm(lines[j].get("text", ""))
        )
        name = _clean_name(block)
        if name:
            return [_span_from_lines(lines, start, end, name)]

    return []


def _extract_from_counsel_list(lines: list[dict]) -> list[dict]:
    for i, line in enumerate(lines):
        raw = line.get("text", "") or ""
        m = _COUNSEL_LIST_HEADING_RE.match(raw)
        if not m:
            continue
        tail = _norm(m.group(1))
        if tail:
            name = _clean_name(tail)
            if name:
                return [_span_from_lines(lines, i, i, name)]

        for j in range(i + 1, len(lines)):
            candidate = _norm(lines[j].get("text", ""))
            if not candidate:
                continue
            if candidate.startswith("*") or candidate.startswith("This summary constitutes"):
                continue
            if candidate.endswith(":"):
                continue
            if candidate and not candidate.lower().startswith("counsel for") and len(candidate) < 120:
                name = _clean_name(candidate)
                if name:
                    return [_span_from_lines(lines, j, j, name)]
            if _MAJOR_STOP_HEADING_RE.match(candidate):
                break
    return []


def rule_first_listed_attorney_representing_appellees(doc: dict) -> list[dict]:
    try:
        lines = _line_dicts(doc)
        spans = _extract_from_standard_counsel(lines)
        if spans:
            return spans
        return _extract_from_counsel_list(lines)
    except Exception:
        return []
