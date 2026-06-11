def rule_page1_company_header_block(doc: dict) -> list[dict]:
    """Match large page-1 company header spans that often contain both state and EIN inline."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and ("Exact name of registrant as specified in its charter" in text or "Exact name of Registrant as specified in its charter" in text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
