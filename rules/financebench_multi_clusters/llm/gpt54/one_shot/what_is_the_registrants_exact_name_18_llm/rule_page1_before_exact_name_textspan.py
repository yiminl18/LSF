def rule_page1_before_exact_name_textspan(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text_span itself starts with the exact-name caption, returning the header."""
    try:
        out = []
        for span in doc.get("texts", []):
            ts = (span.get("text_span", "") or "").lower()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and "exact name of registrant" in ts
            ):
                out.append(span)
        return out
    except Exception:
        return []
