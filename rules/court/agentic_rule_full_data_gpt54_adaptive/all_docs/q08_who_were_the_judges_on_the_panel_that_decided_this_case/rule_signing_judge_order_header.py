import re


_ORDER_RE = re.compile(r"(?i)\b(?:amended\s+order|order)\b")
_SIGNER_RE = re.compile(
    r"^\s*([A-Z][A-Z.\-'\s]+,\s*(?:Chief|Circuit|Senior|District)\s+Judge):\s*$"
)
_STOP_RE = re.compile(
    r"(?i)\b(?:before:|opinion by|summary|counsel)\b"
)
_SKIP_RE = re.compile(r"(?i)\b(?:statement by|dissent by|concurrence by)\b")


def _make_span(text: str, source: dict) -> dict:
    span = {"text": text}
    for field in ("page_no", "line_no", "paragraph_no"):
        if field in source:
            span[field] = source[field]
    return span


def rule_signing_judge_order_header(doc: dict) -> list[dict]:
    try:
        lines = [line for line in (doc.get("lines") or []) if isinstance(line, dict)]
        if not lines:
            return []

        head_lines = lines[:260]
        head_text = "\n".join((line.get("text") or "") for line in head_lines)
        if not _ORDER_RE.search(head_text):
            return []
        if "before:" in head_text.lower():
            return []

        for line in head_lines:
            text = (line.get("text") or "").strip()
            if not text or _STOP_RE.search(text):
                continue
            match = _SIGNER_RE.match(text)
            if match:
                signer_idx = head_lines.index(line)
                window = [text]
                for nxt in head_lines[signer_idx + 1 : signer_idx + 5]:
                    nxt_text = (nxt.get("text") or "").strip()
                    if not nxt_text:
                        if len(window) > 1:
                            break
                        continue
                    if _SKIP_RE.search(nxt_text):
                        continue
                    if _STOP_RE.search(nxt_text):
                        break
                    window.append(nxt_text)
                    if nxt_text.endswith(":") or nxt_text.endswith("."):
                        break
                return [_make_span(" ".join(window), line)]
        return []
    except Exception:
        return []
