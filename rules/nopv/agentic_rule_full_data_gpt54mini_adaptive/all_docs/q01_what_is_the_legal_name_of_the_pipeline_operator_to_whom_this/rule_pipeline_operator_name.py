import re


_HEADER_NOISE_PREFIXES = (
    "u.s. department",
    "of transportation",
    "pipeline and hazardous",
    "materials safety administration",
    "notice of probable violation",
    "proposed compliance order",
    "proposed civil penalty",
    "via electronic mail",
    "via e-mail",
    "certified mail",
    "overnight express delivery",
    "dear ",
    "cpf ",
    "page ",
)

_TITLE_WORDS = (
    "president",
    "vice president",
    "chief executive officer",
    "ceo",
    "manager",
    "director",
    "officer",
    "supervisor",
    "attorney",
    "cfo",
    "coo",
    "vp",
    "v.p.",
    "head",
)

_TITLE_PATTERNS = (
    r"\bpresident\b",
    r"\bvice president\b",
    r"\bchief executive officer\b",
    r"\bceo\b",
    r"\bmanager\b",
    r"\bdirector\b",
    r"\bofficer\b",
    r"\bsupervisor\b",
    r"\battorney\b",
    r"\bcfo\b",
    r"\bcoo\b",
    r"\bvp\b",
    r"\bv\.p\.\b",
    r"\bhead\b",
)

_DESCRIPTOR_WORDS = (
    "subsidiary",
    "division",
    "department",
    "branch",
    "office",
    "operated by",
    "operating by",
    "a subsidiary of",
    "business unit",
    "unit",
)

_COMPANY_PATTERNS = (
    r"\bllc\b",
    r"\bl\.l\.c\.\b",
    r"\blp\b",
    r"\bl\.p\.\b",
    r"\binc\.?\b",
    r"\bcorp\.?\b",
    r"\bcorporation\b",
    r"\bcompany\b",
    r"\bco\.?\b",
    r"\blimited partnership\b",
    r"\bpartnership\b",
    r"\bllp\b",
)

_INDUSTRY_WORDS = (
    "pipeline",
    "pipelines",
    "gas",
    "midstream",
    "energy",
    "transmission",
    "express",
    "utilities",
    "utility",
    "refining",
    "resources",
    "operating",
    "partners",
    "partner",
    "works",
    "storage",
    "services",
    "service",
    "production",
    "oil",
)

_ADDRESS_STREET_RE = re.compile(
    r"(?i)^\s*\d{1,6}\b.*\b(?:street|st\.?|road|rd\.?|avenue|ave\.?|drive|dr\.?|parkway|pkwy\.?|boulevard|blvd\.?|lane|ln\.?|court|ct\.?|circle|cir\.?|highway|hwy\.?|trail|trl\.?|way|suite|ste\.?|floor|fl\.?)\b"
)
_ADDRESS_BOX_RE = re.compile(r"(?i)\b(?:p\.?\s*o\.?\s*box|po box)\b")
_CITY_STATE_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")
_TRAILING_ALIAS_RE = re.compile(r"\s*\((?:[A-Za-z0-9&.,\-/ ]{2,20})\)\s*$")
_DATE_RE = re.compile(
    r"(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b.*\b\d{4}\b"
)
_SENTENCE_NAME_PATTERNS = (
    re.compile(r"(?i)\bproposes? to issue to\s+(?P<name>.+?)\s+a Compliance Order\b"),
    re.compile(r"(?i)\bIt is requested that\s+(?P<name>.+?)\s+maintain\b"),
)
_BODY_NAME_PATTERNS = (
    re.compile(r"(?i)\bnamely,\s+(?P<name>[A-Z][^.;:]+?(?:Co\.,\s*LLC|L\.P\.|LP|LLC|Inc\.|Corp\.|Corporation|Company|Co\.))(?=,|\s+and\b|\.)"),
    re.compile(r"(?i)\boperated by\s+(?P<name>[A-Z][^,.;:]+?)(?=,\s+a subsidiary|,\s+an?\s+subsidiary|,\s+the|,|\.)"),
    re.compile(r"(?i)\binspected(?:\s+the|\s+your)?\s+.*?\bof\s+(?P<name>[A-Z][^.;:]+?(?:Co\.,\s*LLC|L\.P\.|LP|LLC|Inc\.|Corp\.|Corporation|Company|Co\.))(?=’s|'s|\b|,|\.)"),
    re.compile(r"(?i)\binspected(?:\s+the|\s+your)?\s+(?P<name>[A-Z][^.;:]+?(?:Co\.,\s*LLC|L\.P\.|LP|LLC|Inc\.|Corp\.|Corporation|Company|Co\.))(?=’s|'s|\b|,|\.)"),
)


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_header_noise(text: str) -> bool:
    low = text.strip().lower()
    if not low:
        return True
    return any(low.startswith(prefix) for prefix in _HEADER_NOISE_PREFIXES)


def _is_date_line(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return False
    return bool(_DATE_RE.search(raw))


def _extract_name_from_sentence(text: str) -> str | None:
    raw = _normalize_spaces(text)
    for pattern in _SENTENCE_NAME_PATTERNS:
        match = pattern.search(raw)
        if match:
            name = match.group("name")
            name = _TRAILING_ALIAS_RE.sub("", name).strip(" ,;:.")
            return name
    return None


def _extract_name_from_body(lines: list[dict]) -> str | None:
    dear_idx = None
    for idx, item in enumerate(lines):
        text = item.get("text", "").strip()
        if text.lower().startswith("dear "):
            dear_idx = idx
            break
    if dear_idx is None:
        return None

    end_idx = len(lines)
    for idx in range(dear_idx + 1, len(lines)):
        text = lines[idx].get("text", "").strip()
        if re.match(r"^\d+\.\s*$", text) or re.match(r"^\d+\.\s+", text):
            end_idx = idx
            break

    body_text = " ".join(item.get("text", "").strip() for item in lines[dear_idx + 1 : end_idx] if item.get("text", "").strip())
    if not body_text:
        return None

    for pattern in _BODY_NAME_PATTERNS:
        match = pattern.search(body_text)
        if match:
            name = match.group("name")
            name = _TRAILING_ALIAS_RE.sub("", name).strip(" ,;:.")
            return name
    return None


def _is_address_like(text: str) -> bool:
    low = text.strip().lower()
    if not low:
        return False
    if _CITY_STATE_RE.search(text):
        return True
    if _ADDRESS_BOX_RE.search(text):
        return True
    if _ADDRESS_STREET_RE.search(text):
        return True
    if re.search(r"\b(?:street|road|avenue|drive|parkway|boulevard|lane|court|circle|highway|trail|way)\b", low):
        return True
    return False


def _is_person_title(text: str) -> bool:
    low = text.strip().lower()
    if not low:
        return False
    if low.startswith(("mr.", "ms.", "mrs.", "dr.")):
        return True
    return any(re.search(pattern, low) for pattern in _TITLE_PATTERNS)


def _score_candidate(text: str, distance_to_addr: int) -> int:
    raw = _normalize_spaces(text)
    low = raw.lower()
    if not raw:
        return -999
    if _is_header_noise(raw) or _is_address_like(raw) or raw.lower().startswith("dear "):
        return -999

    score = 0
    if any(re.search(pattern, low) for pattern in _COMPANY_PATTERNS):
        score += 6
    if any(word in low for word in _INDUSTRY_WORDS):
        score += 2
    if 2 <= len(raw.split()) <= 8:
        score += 1
    if re.search(r"[A-Z]", raw) and re.search(r"[a-z]", raw):
        score += 1
    if distance_to_addr <= 3:
        score += 4 - distance_to_addr
    if any(word in low for word in _DESCRIPTOR_WORDS):
        score -= 5
    if _is_person_title(raw):
        score -= 8
    if low.startswith(("a subsidiary of", "subsidiary of")):
        score -= 6
    if "@" in raw:
        score -= 10
    if raw.isupper() and len(raw.split()) <= 4:
        score -= 2
    return score


def rule_pipeline_operator_name(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        ordered = sorted(
            (
                l.get("page_no", 0),
                l.get("line_no", 0),
                _normalize_spaces(l.get("text", "")),
            )
            for l in lines
            if _normalize_spaces(l.get("text", ""))
        )
        if not ordered:
            return []

        first_page = ordered[0][0]
        page_lines = [
            {"page_no": p, "line_no": ln, "text": txt}
            for p, ln, txt in ordered
            if p == first_page
        ]
        if not page_lines:
            return []

        body_name = _extract_name_from_body(page_lines)
        if body_name:
            return [
                {
                    "text": body_name,
                    "page_no": page_lines[0]["page_no"],
                    "line_no": page_lines[0]["line_no"],
                }
            ]

        cutoff = len(page_lines)
        for idx, item in enumerate(page_lines):
            if item["text"].strip().upper().startswith("CPF "):
                cutoff = idx
                break

        header_block = page_lines[:cutoff]
        if not header_block:
            header_block = page_lines[: min(len(page_lines), 30)]

        start_idx = 0
        for idx, item in enumerate(header_block):
            if _is_date_line(item["text"]):
                start_idx = idx + 1
                break

        addr_idx = None
        for idx, item in enumerate(header_block[start_idx:], start=start_idx):
            if _is_address_like(item["text"]):
                addr_idx = idx
                break
        if addr_idx is None:
            addr_idx = len(header_block)

        candidates = []
        for idx, item in enumerate(header_block[start_idx:addr_idx], start=start_idx):
            text = item["text"]
            if not text or _is_header_noise(text) or text.lower().startswith("dear "):
                continue
            if _is_person_title(text):
                continue
            score = _score_candidate(text, addr_idx - idx)
            if score > -999:
                candidates.append((score, idx, item))

        if not candidates:
            return []

        candidates.sort(key=lambda x: (x[0], x[1]))
        best_score, best_idx, best = candidates[-1]
        best_text = best["text"]
        best_text = _TRAILING_ALIAS_RE.sub("", best_text).strip()
        sentence_name = _extract_name_from_sentence(best_text)
        if sentence_name:
            best_text = sentence_name

        return [
            {
                "text": best_text,
                "page_no": best["page_no"],
                "line_no": best["line_no"],
            }
        ]
    except Exception:
        return []
