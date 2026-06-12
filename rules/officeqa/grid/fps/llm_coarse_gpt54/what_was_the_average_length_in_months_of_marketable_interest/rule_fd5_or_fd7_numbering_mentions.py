def rule_fd5_or_fd7_numbering_mentions(doc: dict) -> list[dict]:
    """Match spans explicitly numbered FD-5, FD-7, or FO-7 and mentioning average length."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if re.search(r'\b(fd[-\s]?5|fd[-\s]?7|fo[-\s]?7)\b', txt) and "average length" in txt:
                out.append(span)
    except Exception:
        return []
    return out
