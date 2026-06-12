def rule_quarter_receipts_analysis_text(doc: dict) -> list[dict]:
    """Match prose spans discussing quarter receipts and individual income taxes, often containing the answer directly in later-era documents."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "list_item"}:
                continue
            txt = (span.get("text") or "")
            if re.search(r'Individual income taxes', txt, re.I) and re.search(r'quarter|July through September|October-December|first quarter|fourth quarter', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
