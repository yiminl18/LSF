def rule_exhibit_related_on_page_one_8k(doc: dict) -> list[dict]:
    """Match page-1 8-K spans mentioning exhibit references, common in short 8-Ks."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "")
            if re.search(r"\bexhibit\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
