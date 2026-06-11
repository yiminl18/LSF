def rule_page1_first_page_all_ein_state_related(doc: dict) -> list[dict]:
    """Match all first-page spans related to state/EIN by keyword or value pattern."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") != 1:
                continue
            if re.search(r"State or other jurisdiction|Employer Identification|I\.?R\.?S\.?", txt, re.I):
                out.append(span)
            elif re.fullmatch(r"\d{2}-\d{7}", txt):
                out.append(span)
            elif re.fullmatch(r"Delaware|Washington|New York|Jersey(?: \(Channel Islands\))?", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
