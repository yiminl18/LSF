def rule_additional_information_mailing_address_and_executive_offices(doc: dict) -> list[dict]:
    """Match narrative spans stating 'mailing address and executive offices are located at ...'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'mailing address and executive offices are located at', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
