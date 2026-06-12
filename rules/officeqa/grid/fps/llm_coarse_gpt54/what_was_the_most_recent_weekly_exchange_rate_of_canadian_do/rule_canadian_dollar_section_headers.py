def rule_canadian_dollar_section_headers(doc: dict) -> list[dict]:
    """Match section headers explicitly titled CANADIAN DOLLAR POSITIONS."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                txt = (span.get("text") or "").lower()
                if "canadian dollar positions" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
