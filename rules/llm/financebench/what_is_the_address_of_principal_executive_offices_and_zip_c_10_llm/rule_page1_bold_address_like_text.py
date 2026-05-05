def rule_page1_bold_address_like_text(doc: dict) -> list[dict]:
    """Match bold page-1 text spans that look like street addresses with city/state/ZIP."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if any(k in low for k in ["commission", "form 10-k", "exact name of registrant", "documents incorporated"]):
                continue
            has_num = bool(re.search(r"\b\d{1,6}\b", txt))
            has_street_word = bool(re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way|north|south|east|west|riverfront|bowerman|hamilton|rockledge|terry)\b", low))
            has_zip = bool(re.search(r"\b\d{5}(?:-\d{4})?\b", txt))
            has_city_state = bool(re.search(r",[ ]?[A-Z][a-zA-Z .'-]+,?[ ]+(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)*)", txt))
            if (has_num and has_street_word and (has_zip or has_city_state)) or (has_num and has_zip and len(txt) < 140):
                out.append(span)
        return out
    except Exception:
        return []
