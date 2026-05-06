def rule_phone_with_principal_offices_phrase(doc: dict) -> list[dict]:
    """Match spans combining principal executive offices wording with a phone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"principal executive offices", text, re.I) and re.search(r"(\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\+\d{1,3}\s*\d)", text):
                out.append(span)
        return out
    except Exception:
        return []
