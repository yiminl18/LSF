def rule_other_information_executive_offices_sentence(doc: dict) -> list[dict]:
    """Match narrative spans stating 'principal executive offices are located at ...'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'principal executive offices are located at', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
