def rule_other_information_principal_executive_offices(doc: dict) -> list[dict]:
    """Match narrative spans in 'Other Information' that explicitly state principal executive offices are located at an address."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            t = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'other information|additional information', path, re.I):
                if re.search(r'principal executive offices are located at', t, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
