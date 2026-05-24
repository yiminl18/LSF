def rule_first_financial_table_after_item8(doc: dict) -> list[dict]:
    """Match the first table after an Item 8 Financial Statements header."""
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if "item 8" in txt and "financial statements" in txt:
                for j in range(i + 1, len(texts)):
                    if texts[j].get("label") == "table":
                        return [texts[j]]
        return []
    except Exception:
        return []
