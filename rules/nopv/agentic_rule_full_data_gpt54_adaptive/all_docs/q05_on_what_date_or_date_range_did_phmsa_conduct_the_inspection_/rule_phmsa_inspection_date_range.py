import re


_OPENING_CONTEXT_RE = re.compile(
    r"\b(?:PHMSA|Pipeline and Hazardous Materials Safety Administration|pursuant to|acting as an agent)\b",
    re.IGNORECASE,
)

_ACTION_RE = re.compile(
    r"\b(?:inspected|reviewed|conduct|conducted|performed|investigated|responded)\b",
    re.IGNORECASE,
)

_LEAD_PATTERNS = [
    re.compile(
        r"\b((?:From|On|Between|For)\s+.+?)(?=,?\s+(?:(?:a|an|the|one)\s+)?(?:representative|representatives|inspector|inspectors)\b)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b((?:From|On|Between|For)\s+.+?)(?=,?\s+(?:PHMSA(?:['’]s)?|the Pipeline and Hazardous Materials Safety Administration)\b)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(During\s+(?:PHMSA(?:['’]s)?\s+)?(?:on-site\s+)?(?:field\s+)?inspection\s+on\s+.+?)(?=[.;])",
        re.IGNORECASE | re.DOTALL,
    ),
]


def rule_phmsa_inspection_date_range(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def clean(text: str) -> str:
            text = norm(text)
            text = re.sub(r"\s+,", ",", text)
            return text.rstrip(" ,;:")

        def make_span(text: str, src: dict | None = None) -> dict:
            span = {"text": clean(text)}
            if src:
                for key in ("page_no", "paragraph_no", "line_no"):
                    if key in src:
                        span[key] = src[key]
            return span

        def scan(text: str, src: dict | None = None) -> list[dict] | None:
            compact = norm(text)
            if not compact:
                return None

            sentences = [part.strip() for part in re.split(r"(?<=[.?!])\s+(?=[A-Z])", compact) if part.strip()]
            segments = sentences[:] if sentences else []
            segments.append(compact)

            for segment in segments:
                for pat in _LEAD_PATTERNS:
                    match = pat.search(segment)
                    if not match:
                        continue
                    candidate = clean(match.group(1))
                    if segment == compact and len(candidate) > 120:
                        continue
                    start = match.start()
                    window = segment[start : start + 900]
                    if not _OPENING_CONTEXT_RE.search(window):
                        continue
                    if not _ACTION_RE.search(window):
                        continue
                    return [make_span(candidate, src)]
            return None

        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        for item in paragraphs[:18]:
            found = scan(item.get("text") or "", item)
            if found:
                return found

        pages = [item for item in (doc.get("pages") or []) if isinstance(item, dict)]
        for item in pages[:3]:
            found = scan(item.get("text") or "", item)
            if found:
                return found

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        for i in range(min(len(lines), 80)):
            window_lines = lines[i : i + 8]
            if not window_lines:
                continue
            text = "\n".join((line.get("text") or "") for line in window_lines)
            found = scan(text, window_lines[0])
            if found:
                return found

        text = doc.get("text") or ""
        if isinstance(text, str):
            found = scan(text[:12000], None)
            if found:
                return found

        return []
    except Exception:
        return []
