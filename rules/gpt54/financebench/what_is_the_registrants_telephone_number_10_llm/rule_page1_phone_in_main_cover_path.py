def rule_page1_phone_in_main_cover_path(doc: dict) -> list[dict]:
    """Match phone-like spans whose path_text is the main company cover path on page 1."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if path and phone_re.search(text):
                if "|" not in path or path.count("|") <= 1:
                    out.append(span)
        return out
    except Exception:
        return []
