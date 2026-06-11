def rule_page1_10q_company_header_block(doc: dict) -> list[dict]:
    """Match page-1 10-Q company header blocks where state and EIN are split into nearby spans."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "Exact name of registrant as specified in its charter" in (span.get("text_span", "") or "")
            ):
                out.append(span)
        return out
    except Exception:
        return []
