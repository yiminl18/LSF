def rule_page1_h1_registrant_name(doc: dict) -> list[dict]:
    """Match page-1 H1/section_header spans whose text is the registrant name and whose text_span mentions exact name of registrant."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "exact name of registrant" in ((span.get("text_span") or "") + " " + (span.get("text") or "")).lower()
            ):
                out.append(span)
        return out
    except Exception:
        return []
