def rule_page1_cover_span_with_as_of_and_number_of_shares(doc: dict) -> list[dict]:
    """Match cover-page spans containing 'as of' and 'number of shares of common stock outstanding'."""
    try:
        out = []
        for span in doc.get("texts", []):
            t = " ".join((span.get("text") or "").lower().split())
            if span.get("page_no") in (1, 2):
                if "as of" in t and "number of shares of common stock outstanding" in t:
                    out.append(span)
        return out
    except Exception:
        return []
