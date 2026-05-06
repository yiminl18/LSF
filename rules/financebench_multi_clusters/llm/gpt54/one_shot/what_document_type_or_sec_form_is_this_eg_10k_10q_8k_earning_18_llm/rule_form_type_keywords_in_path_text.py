def rule_form_type_keywords_in_path_text(doc: dict) -> list[dict]:
    """Match spans whose path_text contains CURRENT REPORT or a form name."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            path = ((s.get("structure") or {}).get("path_text") or "").upper()
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", path, re.I) or "CURRENT REPORT" in path:
                out.append(s)
        return out
    except Exception:
        return []
