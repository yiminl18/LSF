def rule_page1_section_header_with_text_span_exact_name(doc: dict) -> list[dict]:
    """Match page-1 section_header spans whose text_span includes the exact-name caption or nearby charter metadata."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            ts = span.get("text_span") or ""
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                if "(Exact name of registrant as specified in its charter)" in ts or "State or other jurisdiction of incorporation" in ts:
                    out.append(span)
        return out
    except Exception:
        return []
