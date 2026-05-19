def rule_item8_heading_spans(doc: dict) -> list[dict]:
    """Match section headers for Item 8 / Financial Statements and Supplementary Data."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            text = (span.get("text") or "").lower()
            if "item 8" in text or "financial statements and supplementary data" in text:
                out.append(span)
        return out
    except Exception:
        return []
