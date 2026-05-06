def rule_page1_ein_or_state_labels(doc: dict) -> list[dict]:
    """Match page-1 spans containing the state/jurisdiction or IRS Employer Identification label text."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r"state\s+or\s+other\s+jurisdiction|i\.?r\.?s\.?\s+employer\s+identification|irs\s+employer\s+identification", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
