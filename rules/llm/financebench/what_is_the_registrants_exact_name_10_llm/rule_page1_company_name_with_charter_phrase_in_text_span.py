def rule_page1_company_name_with_charter_phrase_in_text_span(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text_span contains the exact-name charter phrase."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip()
            tspan = (span.get("text_span") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if "exact name of registrant" in tspan:
                out.append(span)
            elif "exact name of registrant" in ((span.get("text") or "").lower()) and txt.lower() != "(exact name of registrant as specified in its charter)":
                out.append(span)
        return out
    except Exception:
        return []
