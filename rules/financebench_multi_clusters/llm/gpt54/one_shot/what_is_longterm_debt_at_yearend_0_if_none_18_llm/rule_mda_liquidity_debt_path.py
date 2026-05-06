def rule_mda_liquidity_debt_path(doc: dict) -> list[dict]:
    """Match spans in MD&A liquidity/capital resources sections mentioning debt."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if (
                ("management's discussion" in path or "liquidity" in path or "capital resources" in path)
                and re.search(r"\bdebt\b|\blong[- ]term debt\b|\bborrowings\b", txt)
            ):
                out.append(span)
    except Exception:
        return []
    return out
