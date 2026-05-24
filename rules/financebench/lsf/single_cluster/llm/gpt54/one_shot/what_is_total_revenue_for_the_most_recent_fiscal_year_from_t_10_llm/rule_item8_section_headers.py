def rule_item8_section_headers(doc: dict) -> list[dict]:
    """Match section headers for Item 8 Financial Statements and Supplementary Data."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if "item 8" in txt and "financial statements" in txt:
                out.append(span)
        return out
    except Exception:
        return []
