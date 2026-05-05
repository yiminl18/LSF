def rule_page1_section_headers_with_exact_name(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text or text_span contains the exact-name-of-registrant phrase."""
    try:
        out = []
        for span in doc.get("texts", []):
            combined = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and span.get("label") == "section_header" and "exact name of registrant" in combined:
                out.append(span)
        return out
    except Exception:
        return []
