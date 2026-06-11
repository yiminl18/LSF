def rule_page1_address_with_company_city_state_zip(doc: dict) -> list[dict]:
    """Match page-1 spans containing a full office address with city/state/ZIP in one line."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+.*,\s*[A-Z][A-Za-z ]+,\s*(?:[A-Z]{2}|[A-Za-z ]+)\s+\d{5}(?:-\d{4})?\b', t):
                out.append(span)
            elif re.search(r'\bone [A-Za-z].*,\s*[A-Z][A-Za-z ]+,\s*(?:[A-Z]{2}|[A-Za-z ]+)\s+\d{5}(?:-\d{4})?\b', t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
