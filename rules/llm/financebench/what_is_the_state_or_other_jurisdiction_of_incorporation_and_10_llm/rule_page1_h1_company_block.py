def rule_page1_h1_company_block(doc: dict) -> list[dict]:
    """Match the large company-name cover-page block on page 1 that often contains the answer in text_span."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "exact name of registrant" in (((span.get("text_span") or "") + " " + (span.get("text") or "")).lower())
            ):
                out.append(span)
        return out
    except Exception:
        return []
