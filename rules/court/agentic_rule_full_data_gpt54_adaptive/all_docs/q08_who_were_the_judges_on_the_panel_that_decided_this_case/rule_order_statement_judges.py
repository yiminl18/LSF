import re


_SIGNER_RE = re.compile(
    r"^\s*([A-Z][A-Z.\-'\s]+,\s*(?:Chief|Circuit|Senior|District)\s+Judge):\s*$"
)
_STATEMENT_LINE_RE = re.compile(r"(?i)\bStatement by Judge ([A-Za-z.\-]+)\b")


def _make_span(text: str, source: dict) -> dict:
    span = {"text": text}
    for field in ("page_no", "line_no", "paragraph_no"):
        if field in source:
            span[field] = source[field]
    return span


def rule_order_statement_judges(doc: dict) -> list[dict]:
    try:
        lines = [line for line in (doc.get("lines") or []) if isinstance(line, dict)]
        if not lines:
            return []

        head_lines = lines[:120]
        signer = None
        signer_line = None
        for line in head_lines:
            text = (line.get("text") or "").strip()
            match = _SIGNER_RE.match(text)
            if match:
                signer = match.group(1)
                signer_line = line
                break
        if not signer or signer_line is None:
            return []

        for line in head_lines:
            text = (line.get("text") or "").strip()
            matches = _STATEMENT_LINE_RE.findall(text)
            if not matches:
                continue
            names = [f"Judge {name}" for name in matches]
            if len(names) == 1:
                combined = f"{signer}; {names[0]}"
            elif len(names) == 2:
                combined = f"{signer}; {names[0]}; {names[1]}"
            else:
                combined = f"{signer}; " + "; ".join(names)
            return [_make_span(combined, signer_line)]
        return []
    except Exception:
        return []
