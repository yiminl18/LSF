def rule_item7_mda_debt_note_reference(doc: dict) -> list[dict]:
    """Match MDA/financial condition spans that mention long-term debt, often near the answer or note reference."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure") or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"management.?s discussion|financial condition|results of operations|liquidity", path, re.I):
                if re.search(r"\blong[\-\s]?term debt\b|\bdebt\b|\bborrowings\b", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
