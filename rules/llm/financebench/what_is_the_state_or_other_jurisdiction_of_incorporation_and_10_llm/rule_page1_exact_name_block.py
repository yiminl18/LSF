def rule_page1_exact_name_block(doc: dict) -> list[dict]:
    """Match page-1 spans near the registrant cover block containing the exact-name caption."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and "exact name of registrant" in text.lower():
                out.append(span)
        return out
    except Exception:
        return []
