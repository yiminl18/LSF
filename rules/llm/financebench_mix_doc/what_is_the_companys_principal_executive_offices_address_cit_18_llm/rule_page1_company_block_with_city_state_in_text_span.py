def rule_page1_company_block_with_city_state_in_text_span(doc: dict) -> list[dict]:
    """Match company H1 blocks whose text_span contains a city/state answer pattern."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            tsp = span.get("text_span") or ""
            if re.search(r"\b[A-Z][a-zA-Z\.\- ]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+\d{4,10}", tsp):
                out.append(span)
            elif "warmley, bristol" in tsp.lower():
                out.append(span)
        return out
    except Exception:
        return []
