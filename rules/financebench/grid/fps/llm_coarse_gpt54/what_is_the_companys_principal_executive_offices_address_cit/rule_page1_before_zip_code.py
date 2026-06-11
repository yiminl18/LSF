def rule_page1_before_zip_code(doc: dict) -> list[dict]:
    """Match page-1 address-like spans that appear just before a zip code label or zip-only span."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            if span.get("page_no") != 1 or nxt.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            nxt_all = ((nxt.get("text") or "") + " " + (nxt.get("text_span") or "")).strip()
            if re.search(r'zip code', nxt_all, re.I) or re.fullmatch(r'\d{5}(?:-\d{4})?', (nxt.get("text") or "").strip()):
                if re.search(r'\d{1,6}\s+\S+', text) and (
                    re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', text) or
                    re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+', text) or
                    re.search(r'\bUnited Kingdom\b', text)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
