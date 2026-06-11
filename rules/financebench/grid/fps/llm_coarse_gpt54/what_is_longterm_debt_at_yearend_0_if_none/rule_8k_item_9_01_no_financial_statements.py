def rule_8k_item_9_01_no_financial_statements(doc: dict) -> list[dict]:
    """Match 8-K spans stating financial statements are not included or will be filed later, indicating no year-end debt answer in-document."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"financial statements.*not included", txt, re.I):
                out.append(span)
            elif re.search(r"pro forma financial information.*not included", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
