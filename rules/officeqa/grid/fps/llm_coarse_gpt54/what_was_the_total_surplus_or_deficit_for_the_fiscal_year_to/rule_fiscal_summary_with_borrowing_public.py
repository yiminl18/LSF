def rule_fiscal_summary_with_borrowing_public(doc: dict) -> list[dict]:
    """Match tables containing total surplus/deficit and borrowing from the public."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'total surplus.*deficit', txt, re.I) and re.search(r'borrowing from the public', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
