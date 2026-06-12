def rule_contains_total_surplus_plus_or_deficit_minus(doc: dict) -> list[dict]:
    """Match tables using the exact phrase 'Total surplus (+) or deficit (-)'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'total surplus\s*\(\+\)\s*or deficit\s*\(-\)', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
