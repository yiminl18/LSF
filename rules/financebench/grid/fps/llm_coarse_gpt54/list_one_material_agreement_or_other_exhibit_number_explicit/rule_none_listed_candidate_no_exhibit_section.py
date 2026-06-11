def rule_none_listed_candidate_no_exhibit_section(doc: dict) -> list[dict]:
    """Return exhibit-related headers/tables; if none exist, supports a 'none listed' conclusion."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            blob = (span.get("text", "") or "") + " " + ((span.get("structure", {}) or {}).get("path_text", "") or "")
            if re.search(r'exhibit|item\s*9\.01|item\s*15|exhibit index', blob, re.I):
                out.append(span)
    except Exception:
        return []
    return out
