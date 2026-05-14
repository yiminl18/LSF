def rule_page1_cover_span_with_as_of_and_issued_outstanding(doc: dict) -> list[dict]:
    """Match cover-page spans containing 'as of' and 'issued and outstanding'."""
    try:
        out = []
        for span in doc.get("texts", []):
            t = " ".join((span.get("text") or "").lower().split())
            if span.get("page_no") in (1, 2):
                if "as of" in t and "issued and outstanding" in t:
                    out.append(span)
        return out
    except Exception:
        return []
