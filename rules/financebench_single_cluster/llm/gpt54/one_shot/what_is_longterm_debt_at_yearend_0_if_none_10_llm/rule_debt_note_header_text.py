def rule_debt_note_header_text(doc: dict) -> list[dict]:
    """Match debt-related section headers themselves as anchors for nearby answer spans."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt) or "notes payable" in txt:
                out.append(span)
        return out
    except Exception:
        return []
