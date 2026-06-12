def rule_table_with_amounts_outstanding(doc: dict) -> list[dict]:
    """Match tables containing amounts outstanding/in circulation wording."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'amounts?\s+outstanding', txt, re.I):
                out.append(span)
            elif re.search(r'in\s+circulation', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
