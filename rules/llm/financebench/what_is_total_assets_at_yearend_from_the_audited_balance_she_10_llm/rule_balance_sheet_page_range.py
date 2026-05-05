def rule_balance_sheet_page_range(doc: dict) -> list[dict]:
    """Match tables on mid/late pages where audited financial statements usually appear."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and 20 <= int(span.get("page_no", 0)) <= 120:
                text = (span.get("text") or "").lower()
                if "assets" in text and ("liabilities" in text or "equity" in text):
                    out.append(span)
        return out
    except Exception:
        return []
