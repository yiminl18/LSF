def rule_page1_company_header_block(doc: dict) -> list[dict]:
    """Match large page-1 company header blocks that often contain both state and EIN inline."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and len(text) > 3
                and any(c.isalpha() for c in text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
