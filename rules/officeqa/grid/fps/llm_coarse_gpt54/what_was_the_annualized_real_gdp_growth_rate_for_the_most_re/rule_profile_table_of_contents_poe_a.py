def rule_profile_table_of_contents_poe_a(doc: dict) -> list[dict]:
    """Match table-of-contents entries for Profile of the Economy GDP chart/table references."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Contents" in path and re.search(r"POE[-\s]?A.*Growth of real gross domestic product", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
