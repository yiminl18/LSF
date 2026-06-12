def rule_ffo2_section_header(doc: dict) -> list[dict]:
    """Match section headers naming Table FFO-2 / Budget Receipts by Source."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") == "section_header":
                if re.search(r'FFO-2', txt, re.I) or re.search(r'Budget Receipts by Source', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
