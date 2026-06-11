def rule_page1_8k_company_header_block(doc: dict) -> list[dict]:
    """Match page-1 8-K company header blocks that often inline state, file number, and EIN."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "Exact name of registrant as specified in its charter" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
