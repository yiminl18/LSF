def rule_form_header_page1(doc: dict) -> list[dict]:
    """Match top-of-document FORM 10-K/10-Q/8-K headers on page 1."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip().upper()
            if span.get("page_no") == 1 and span.get("label") == "section_header" and txt in {"FORM 10-K", "FORM 10-Q", "FORM 8-K"}:
                out.append(span)
        return out
    except Exception:
        return []
