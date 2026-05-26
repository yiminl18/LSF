import re


_ANCHOR_RE = re.compile(
    r"^\s*(?:ORDER(?:\s+AND\s+AMENDED)?|AMENDED\s+OPINION|OPINION)\s*$",
    re.IGNORECASE,
)
_SIGNATURE_RE = re.compile(
    r"^\s*[A-Z][A-Z0-9 .,'’&\-()]+,\s+(?:Circuit|District|Senior|Chief)\s+Judge:\s*$"
)
_PER_CURIAM_RE = re.compile(r"^\s*PER CURIAM:?\s*$", re.IGNORECASE)


def rule_majority_author_signature(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def span_with_meta(item: dict, text: str | None = None) -> dict:
            span = {"text": text if text is not None else norm(item.get("text") or "")}
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in item:
                    span[key] = item[key]
            return span

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if lines:
            anchor_idx = 0
            for idx, item in enumerate(lines[:500]):
                if _ANCHOR_RE.match(norm(item.get("text") or "")):
                    anchor_idx = idx
                    break

            for item in lines[anchor_idx : min(len(lines), anchor_idx + 260)]:
                text = norm(item.get("text") or "")
                if not text:
                    continue
                if _PER_CURIAM_RE.match(text):
                    return [span_with_meta(item, "PER CURIAM authored the majority opinion.")]
                if _SIGNATURE_RE.match(text):
                    return [span_with_meta(item, text)]

        for key, limit in (("paragraphs", 160), ("pages", 20)):
            items = [item for item in (doc.get(key) or []) if isinstance(item, dict)]
            for item in items[:limit]:
                text = norm(item.get("text") or "")
                if not text:
                    continue
                if _PER_CURIAM_RE.match(text):
                    return [span_with_meta(item, "PER CURIAM authored the majority opinion.")]
                if _SIGNATURE_RE.match(text):
                    return [span_with_meta(item, text)]

        text = doc.get("text") or ""
        lines_text = text.splitlines()
        anchor_idx = 0
        for idx, line in enumerate(lines_text[:500]):
            if _ANCHOR_RE.match(norm(line)):
                anchor_idx = idx
                break

        for line in lines_text[anchor_idx : anchor_idx + 260]:
            clean = norm(line)
            if not clean:
                continue
            if _PER_CURIAM_RE.match(clean):
                return [{"text": "PER CURIAM authored the majority opinion."}]
            if _SIGNATURE_RE.match(clean):
                return [{"text": clean}]

        return []
    except Exception:
        return []
