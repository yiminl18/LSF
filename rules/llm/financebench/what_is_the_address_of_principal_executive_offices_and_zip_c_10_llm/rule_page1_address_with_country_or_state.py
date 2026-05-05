def rule_page1_address_with_country_or_state(doc: dict) -> list[dict]:
    """Match page-1 address-like spans ending with a country, state name, or state abbreviation."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\b\d{1,6}\b", txt) and (
                re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", txt)
                or re.search(r"\b(new jersey|new york|oregon|washington|maryland|california|illinois|united kingdom|australia)\b", low)
            ):
                out.append(span)
        return out
    except Exception:
        return []
