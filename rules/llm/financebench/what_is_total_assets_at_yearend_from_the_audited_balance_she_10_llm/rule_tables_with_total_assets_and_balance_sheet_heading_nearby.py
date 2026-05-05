def rule_tables_with_total_assets_and_balance_sheet_heading_nearby(doc: dict) -> list[dict]:
    """Match tables with total assets and nearby balance sheet heading within a short window."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            if "total assets" not in (span.get("text") or "").lower():
                continue
            nearby = texts[max(0, i - 8):i + 1]
            nearby_text = " ".join((x.get("text") or "").lower() for x in nearby)
            if "balance sheet" in nearby_text or "financial position" in nearby_text:
                out.append(span)
        return out
    except Exception:
        return []
