def rule_tables_with_fd1_and_debt_held_by_public(doc: dict) -> list[dict]:
    """Match FD-1 tables that also mention debt held by the public, common in later issues."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'fd[\-–— ]?1|summary of federal debt', text, re.I) and re.search(r'debt held by the public|held by the public', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
