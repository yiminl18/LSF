def rule_page1_span_before_exact_name_parenthetical_same_header(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text_span contains the exact-name parenthetical, implying the header text is the company name."""
    try:
        texts = doc.get("texts", [])
        out = []
        marker = "(Exact name of registrant as specified in its charter)"
        for span in texts:
            if span.get("page_no") == 1 and marker in (span.get("text_span") or ""):
                out.append(span)
        return out
    except Exception:
        return []
