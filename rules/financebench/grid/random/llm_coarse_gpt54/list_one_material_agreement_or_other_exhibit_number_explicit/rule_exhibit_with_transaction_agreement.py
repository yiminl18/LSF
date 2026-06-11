def rule_exhibit_with_transaction_agreement(doc: dict) -> list[dict]:
    """Match spans containing an exhibit number near 'Transaction Agreement'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"\bExhibit\s+\d+(?:\.\d+)?[A-Za-z]?\b", txt, re.I) and re.search(r"\btransaction agreement\b", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
