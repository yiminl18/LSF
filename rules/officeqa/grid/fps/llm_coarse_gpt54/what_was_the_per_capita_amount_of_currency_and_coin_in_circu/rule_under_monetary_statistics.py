def rule_under_monetary_statistics(doc: dict) -> list[dict]:
    """Match spans under the Monetary Statistics section mentioning currency and coin in circulation."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if re.search(r'monetary\s+statistics', path, re.I) and re.search(r'currency\s+and\s+coin', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
