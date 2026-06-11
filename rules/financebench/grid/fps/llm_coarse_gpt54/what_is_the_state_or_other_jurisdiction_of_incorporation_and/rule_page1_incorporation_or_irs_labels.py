def rule_page1_incorporation_or_irs_labels(doc: dict) -> list[dict]:
    """Match page-1 spans containing incorporation or IRS Employer Identification label text."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", txt, re.I):
                out.append(span)
            elif re.search(r"(i\.?r\.?s\.?\s+)?employer\s+identification", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
