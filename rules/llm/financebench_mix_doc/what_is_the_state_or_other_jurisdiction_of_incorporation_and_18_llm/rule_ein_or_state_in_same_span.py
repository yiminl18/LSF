def rule_ein_or_state_in_same_span(doc: dict) -> list[dict]:
    """Match spans containing both a state/jurisdiction label and an EIN or IRS label."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            has_state = re.search(r"state or other jurisdiction of incorporation|state or other jurisdiction of incorporation or organization|state or other jurisdiction of incorporation or organization", text, re.I)
            has_irs = re.search(r"i\.?r\.?s\.? employer identification|employer identification no", text, re.I)
            has_ein = re.search(r"\b\d{2}-\d{7}\b", text)
            if has_state and (has_irs or has_ein):
                out.append(span)
        return out
    except Exception:
        return []
