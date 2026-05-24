def rule_debt_in_consolidated_statements_path(doc: dict) -> list[dict]:
    """Match spans in consolidated statements paths that mention debt."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "consolidated" in path and (re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt)):
                out.append(span)
        return out
    except Exception:
        return []
