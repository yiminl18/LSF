def rule_label_text_near_irs_phrase(doc: dict) -> list[dict]:
    """Match text spans whose text itself contains the IRS phrase or EIN number."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"i\.?r\.?s\.? employer identification|employer identification no|\b\d{2}-\d{7}\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
