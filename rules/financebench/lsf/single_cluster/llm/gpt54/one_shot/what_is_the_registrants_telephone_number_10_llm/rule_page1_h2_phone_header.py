def rule_page1_h2_phone_header(doc: dict) -> list[dict]:
    """Match page-1 H2 spans whose header text is or contains the phone number."""
    import re
    try:
        out = []
        phone_re = re.compile(r"^\s*(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\d{3}-\d{3}-\d{4}|\d{3}-\d{4}-\d{4})\s*$")
        for span in doc.get("texts", []):
            level = ((span.get("structure") or {}).get("level") or "")
            if span.get("page_no") == 1 and level == "H2":
                text = (span.get("text") or "").strip()
                combo = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if phone_re.search(text) or "telephone number" in combo.lower():
                    out.append(span)
        return out
    except Exception:
        return []
