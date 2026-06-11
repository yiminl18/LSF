def rule_additional_information_mailing_address(doc: dict) -> list[dict]:
    """Match narrative spans in 'Additional Information' that state mailing address and executive offices."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            t = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'additional information', path, re.I):
                if re.search(r'mailing address and executive offices are located at', t, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
