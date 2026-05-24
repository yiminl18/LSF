def rule_tables_with_assets_and_balance_sheet_title_in_text(doc: dict) -> list[dict]:
    """Match tables whose own text includes balance sheet title and assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if ("balance sheet" in low or "financial position" in low) and "assets" in low:
                out.append(span)
        return out
    except Exception:
        return []
