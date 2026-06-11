def rule_narrative_address_located_at(doc: dict) -> list[dict]:
    """Match narrative spans anywhere that say offices are located at a specific address."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            t = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'(principal executive offices|mailing address and executive offices|executive offices)\s+are located at', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
