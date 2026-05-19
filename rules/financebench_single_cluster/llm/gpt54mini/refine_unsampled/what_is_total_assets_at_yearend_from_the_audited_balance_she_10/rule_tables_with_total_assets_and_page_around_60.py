def rule_tables_with_total_assets_and_page_around_60(doc: dict) -> list[dict]:
    """Match total-assets tables around page 60, common in longer filings."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and 58 <= int(span.get("page_no", -1)) <= 62:
                if "total assets" in (span.get("text") or "").lower() or "assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
