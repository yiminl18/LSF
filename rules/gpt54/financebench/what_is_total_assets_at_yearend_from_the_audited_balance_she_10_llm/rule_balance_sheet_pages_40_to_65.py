def rule_balance_sheet_pages_40_to_65(doc: dict) -> list[dict]:
    """Match tables on the especially common page range where audited balance sheets appear."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and 40 <= int(span.get("page_no", -1)) <= 65:
                text = (span.get("text") or "").lower()
                if "assets" in text or "liabilities" in text or "equity" in text:
                    out.append(span)
        return out
    except Exception:
        return []
