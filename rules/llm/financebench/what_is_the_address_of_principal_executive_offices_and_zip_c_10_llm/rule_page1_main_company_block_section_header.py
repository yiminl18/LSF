def rule_page1_main_company_block_section_header(doc: dict) -> list[dict]:
    """Match large company-name section headers on page 1 whose text_span often contains the address."""
    try:
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").strip()
            tsp = (span.get("text_span") or "").strip()
            if span.get("bold") != 1:
                continue
            if len(txt) < 3:
                continue
            if any(k in txt.lower() for k in ["form 10-k", "commission", "annual report", "transition report"]):
                continue
            if tsp:
                out.append(span)
        return out
    except Exception:
        return []
