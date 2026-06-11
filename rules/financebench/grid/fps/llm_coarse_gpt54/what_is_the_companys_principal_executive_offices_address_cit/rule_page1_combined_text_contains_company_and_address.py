def rule_page1_combined_text_contains_company_and_address(doc: dict) -> list[dict]:
    """Match page-1 large company block spans whose combined text includes the principal office address."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                if re.search(r'address of principal executive offices', txt, re.I) and re.search(r'\d{1,6}\s+\S+', txt):
                    out.append(span)
        return out
    except Exception:
        return []
