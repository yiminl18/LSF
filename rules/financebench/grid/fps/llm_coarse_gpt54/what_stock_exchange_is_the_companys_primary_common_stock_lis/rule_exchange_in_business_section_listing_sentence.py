def rule_exchange_in_business_section_listing_sentence(doc: dict) -> list[dict]:
    """Match business-section sentences that state where common stock is listed."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "")
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'item 1|business', path, re.I) and re.search(r'common stock .* (listed|trades) on .*?(new york stock exchange|nasdaq)', combined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
