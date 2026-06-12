def rule_federal_debt_statutory_limit_table(doc: dict) -> list[dict]:
    """Match table spans in the Federal Debt section mentioning debt subject to statutory limit or statutory limitation."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "federal debt" in path.lower() and re.search(r"statutory (limit|limitation)|debt subject to statutory", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
