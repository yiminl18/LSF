def rule_modern_financial_operations_receipts_tables(doc: dict) -> list[dict]:
    """Match receipt tables under Financial Operations in modern bulletins."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if span.get("label") == "table":
                if re.search(r'Financial Operations', path, re.I) and re.search(r'Receipts by Source|Individual', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
