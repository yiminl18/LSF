import re


_TOKEN_RE = r"(?:[A-Z]\.|[A-Z]{2,}|[A-Z][A-Za-z'’.\-]*)"
_NAME_RE = re.compile(
    rf"^(?:{_TOKEN_RE})(?:\s+{_TOKEN_RE}){{0,5}}(?:,\s*(?:Jr\.|Sr\.|II|III|IV))?$",
    re.I,
)

_STOP_RE = re.compile(
    r"(?i)\b(?:opinion by|summary|counsel|statement by|dissent by|"
    r"concurrence by|concurring by|order|petition|filed|argued|submitted)\b"
)
_ROLE_RE = re.compile(
    r"(?i)\b(?:united states\s+|u\.s\.\s+)?(?:chief|senior)\s+judge\b|"
    r"\b(?:united states\s+|u\.s\.\s+)?(?:district|magistrate|bankruptcy)\s+judge\b|"
    r"\bcircuit\s+judges?\b"
)
_NOISE_RE = re.compile(
    r"(?i)\b(?:judge|judges|circuit|district|magistrate|bankruptcy|"
    r"opinion|summary|counsel|before|statement|dissent|concurrence|"
    r"argued|submitted|filed|panel|sitting by designation)\b"
)


def _as_int(value):
    try:
        return int(value)
    except Exception:
        return value


def _sorted_items(doc: dict) -> list[dict]:
    lines = doc.get("lines") or []
    if isinstance(lines, list) and lines:
        return sorted(
            [line for line in lines if isinstance(line, dict)],
            key=lambda x: (_as_int(x.get("page_no", 0)), _as_int(x.get("line_no", 0))),
        )

    paragraphs = doc.get("paragraphs") or []
    if isinstance(paragraphs, list) and paragraphs:
        return sorted(
            [para for para in paragraphs if isinstance(para, dict)],
            key=lambda x: (_as_int(x.get("page_no", 0)), _as_int(x.get("paragraph_no", 0))),
        )

    pages = doc.get("pages") or []
    if isinstance(pages, list) and pages:
        out = []
        for page in sorted([p for p in pages if isinstance(p, dict)], key=lambda x: _as_int(x.get("page_no", 0)))[:5]:
            page_no = page.get("page_no")
            for idx, raw_line in enumerate((page.get("text") or "").splitlines(), start=1):
                out.append({"page_no": page_no, "line_no": idx, "text": raw_line})
        return out

    return []


def _make_span(text: str, source: dict) -> dict:
    span = {"text": text}
    for field in ("page_no", "line_no", "paragraph_no"):
        if field in source:
            span[field] = source[field]
    return span


def _format_names(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _parse_panel_names(fragment: str) -> list[str]:
    text = re.sub(r"\s+", " ", (fragment or "").replace("\u00a0", " ")).strip()
    if not text:
        return []

    text = re.sub(r"(?i)\b(?:united states\s+|u\.s\.\s+)?(?:chief|senior)\s+judge(?:s)?\b", " ", text)
    text = re.sub(r"(?i)\b(?:united states\s+|u\.s\.\s+)?(?:district|magistrate|bankruptcy)\s+judge(?:s)?\b", " ", text)
    text = re.sub(r"(?i)\bcircuit\s+judges?\b", " ", text)
    text = re.sub(r"(?i)\bsitting by designation\b", " ", text)

    suffix_map = {}

    def protect_suffix(match):
        placeholder = f"__SUF_{len(suffix_map)}__"
        suffix_map[placeholder] = match.group(1)
        return placeholder

    text = re.sub(r",\s*(Jr\.?|Sr\.?|II|III|IV)(?=[\s,.;:*]|$)", protect_suffix, text, flags=re.I)
    text = re.sub(r"\s+(?:and|&)\s+", ", ", text)
    text = text.replace(";", ",")
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)

    names = []
    seen = set()
    for chunk in text.split(","):
        candidate = re.sub(r"\s+", " ", chunk).strip(" \t\r\n .;:*")
        for placeholder, suffix in suffix_map.items():
            candidate = candidate.replace(placeholder, f", {suffix}")
        if not candidate:
            continue
        candidate = re.sub(r"(?i)^(?:the\s+honorable\s+|hon\.?\s+)", "", candidate).strip()
        if not candidate or _NOISE_RE.search(candidate):
            continue
        if not _NAME_RE.match(candidate):
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(candidate)
    return names


def rule_panel_judges_before_header(doc: dict) -> list[dict]:
    try:
        items = _sorted_items(doc)
        if not items:
            full_text = re.sub(r"\s+", " ", doc.get("text") or "").strip()
            if not full_text:
                return []
            m = re.search(r"(?i)\bbefore:\s*(.+)", full_text)
            if not m:
                return []
            fragment = m.group(1)
            role_matches = list(_ROLE_RE.finditer(fragment))
            if role_matches:
                fragment = fragment[: role_matches[-1].end()]
            else:
                stop = _STOP_RE.search(fragment)
                if stop:
                    fragment = fragment[: stop.start()]
            names = _parse_panel_names(fragment)
            if names:
                text = _format_names(names)
                return [{"text": text}]
            return [{"text": fragment.strip(" \t\r\n .;:*")}] if fragment.strip() else []

        for idx, item in enumerate(items[:400]):
            text = re.sub(r"\s+", " ", (item.get("text") or "").replace("\u00a0", " ")).strip()
            if not text or "before:" not in text.lower():
                continue

            window = [text]
            for nxt in items[idx + 1 : idx + 5]:
                nxt_text = re.sub(r"\s+", " ", (nxt.get("text") or "").replace("\u00a0", " ")).strip()
                if not nxt_text:
                    continue
                window.append(nxt_text)
                if _STOP_RE.search(nxt_text):
                    break

            fragment = " ".join(window)
            fragment = re.sub(r"(?i)^.*?\bbefore:\s*", "", fragment, count=1)

            role_matches = list(_ROLE_RE.finditer(fragment))
            if role_matches:
                fragment = fragment[: role_matches[-1].end()]
            else:
                stop = _STOP_RE.search(fragment)
                if stop:
                    fragment = fragment[: stop.start()]

            names = _parse_panel_names(fragment)
            if names:
                text = _format_names(names)
                return [_make_span(text, item)]

            fallback = fragment.strip(" \t\r\n .;:*")
            if fallback:
                return [_make_span(fallback, item)]

        return []
    except Exception:
        return []
