def rule_page1_span_with_principal_executive_offices_in_text(doc: dict) -> list[dict]:
    """Match page 1 spans whose own text contains the principal executive offices phrase and address content."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = span.get("text") or ""
            low = txt.lower()
            if span.get("page_no") == 1 and "principal executive offices" in low:
                out.append(span)
        return out
    except Exception:
        return []
