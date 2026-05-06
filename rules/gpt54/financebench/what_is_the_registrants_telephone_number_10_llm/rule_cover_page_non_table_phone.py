def rule_cover_page_non_table_phone(doc: dict) -> list[dict]:
    """Match non-table cover-page spans with phone-like numbers, excluding later business tables."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}-\d{3}-\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") == "table":
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
