def rule_page1_non_table_phone(doc: dict) -> list[dict]:
    """Match non-table page-1 spans with phone-like patterns."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") != "table":
                text = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if phone_re.search(text):
                    out.append(span)
        return out
    except Exception:
        return []
