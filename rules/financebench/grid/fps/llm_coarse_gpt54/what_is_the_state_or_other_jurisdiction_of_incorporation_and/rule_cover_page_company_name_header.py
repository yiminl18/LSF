def rule_cover_page_company_name_header(doc: dict) -> list[dict]:
    """Match the page-1 company-name header under which incorporation state and EIN usually appear."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 and span.get("page_no") != 2:
                continue
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").strip()
            if any(k in txt.lower() for k in ["inc.", "incorporated", "corporation", "company", "plc"]):
                if "commission" not in txt.lower() and "form 8-k" not in txt.lower() and "form 10-k" not in txt.lower() and "form 10-q" not in txt.lower():
                    out.append(span)
        return out
    except Exception:
        return []
