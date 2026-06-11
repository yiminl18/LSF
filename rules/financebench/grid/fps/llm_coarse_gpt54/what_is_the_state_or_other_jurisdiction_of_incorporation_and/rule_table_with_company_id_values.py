def rule_table_with_company_id_values(doc: dict) -> list[dict]:
    """Match page-1 tables that contain both a state/jurisdiction value and an EIN-like value."""
    try:
        import re
        states = r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)"
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table" or span.get("page_no") != 1:
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            joined = " ".join((c.get("text") or "") for c in cells)
            if re.search(states, joined, re.I) and re.search(r"\d{2}-\d{7}", joined):
                out.append(span)
        return out
    except Exception:
        return []
