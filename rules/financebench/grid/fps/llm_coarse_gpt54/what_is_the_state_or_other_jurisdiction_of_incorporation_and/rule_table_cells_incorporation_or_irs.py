def rule_table_cells_incorporation_or_irs(doc: dict) -> list[dict]:
    """Match table spans on page 1 whose cells mention incorporation or IRS Employer Identification."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            if span.get("page_no") != 1:
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            joined = " ".join((c.get("text") or "") for c in cells)
            if re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", joined, re.I):
                out.append(span)
            elif re.search(r"(i\.?r\.?s\.?\s+)?employer\s+identification", joined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
