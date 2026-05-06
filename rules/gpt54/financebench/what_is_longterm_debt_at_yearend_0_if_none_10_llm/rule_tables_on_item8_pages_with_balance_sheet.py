def rule_tables_on_item8_pages_with_balance_sheet(doc: dict) -> list[dict]:
    """Match tables on pages near Item 8 that contain balance sheet or long-term debt text."""
    try:
        out = []
        for span in rule_pages_near_item_8_from_toc(doc):
            txt = (span.get("text") or "").lower()
            if "balance sheet" in txt or "long-term debt" in txt or "long term debt" in txt:
                out.append(span)
        return out
    except Exception:
        return []
