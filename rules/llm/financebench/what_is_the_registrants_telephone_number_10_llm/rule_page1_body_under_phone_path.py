def rule_page1_body_under_phone_path(doc: dict) -> list[dict]:
    """Match body spans whose path_text itself is a phone number or contains one."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if span.get("page_no") == 1 and (span.get("structure") or {}).get("level") == "Body":
                if phone_re.search(path):
                    out.append(span)
        return out
    except Exception:
        return []
