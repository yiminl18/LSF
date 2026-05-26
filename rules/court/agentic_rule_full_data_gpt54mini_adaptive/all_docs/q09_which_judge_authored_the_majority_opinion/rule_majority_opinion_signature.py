import re


def rule_majority_opinion_signature(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def with_meta(item: dict) -> dict:
            span = {"text": (item.get("text") or "").strip()}
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in item:
                    span[key] = item[key]
            return span

        signature_re = re.compile(
            r"^\s*[A-Z][A-Z0-9 .,'’&\-()]+,\s+(?:Circuit|District|Senior|Chief)\s+Judge(?:,.*)?\s*:\s*$"
        )
        per_curiam_re = re.compile(r"^\s*PER CURIAM:?\s*$", re.IGNORECASE)

        for key in ("lines", "paragraphs", "pages"):
            items = [item for item in (doc.get(key) or []) if isinstance(item, dict)]
            for item in items:
                text = norm(item.get("text"))
                if not text:
                    continue
                if per_curiam_re.match(text):
                    return [with_meta(item)]
                if signature_re.match(text):
                    return [with_meta(item)]

        full_text = doc.get("text") or ""
        m = re.search(
            r"^\s*[A-Z][A-Z0-9 .,'’&\-()]+,\s+(?:Circuit|District|Senior|Chief)\s+Judge(?:,.*)?\s*:\s*$",
            full_text,
            flags=re.MULTILINE,
        )
        if m:
            return [{"text": m.group(0).strip()}]

        m = re.search(r"^\s*PER CURIAM:?\s*$", full_text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return [{"text": m.group(0).strip()}]

        return []
    except Exception:
        return []
