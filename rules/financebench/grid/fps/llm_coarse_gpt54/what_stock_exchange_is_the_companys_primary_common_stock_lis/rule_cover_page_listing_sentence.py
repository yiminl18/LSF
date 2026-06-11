def rule_cover_page_listing_sentence(doc: dict) -> list[dict]:
    """Match cover-page narrative sentences stating the common stock is listed on an exchange."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if span.get("page_no") <= 5 and re.search(r'common stock .* listed on .*?(new york stock exchange|nasdaq)', combined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
