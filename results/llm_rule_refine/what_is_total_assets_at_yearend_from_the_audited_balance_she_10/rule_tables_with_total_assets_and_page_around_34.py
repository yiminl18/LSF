def rule_tables_with_total_assets_and_page_around_34(doc: dict) -> list[dict]:
    """Match total-assets tables around page 34, common in some retail/tech filings."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and 33 <= int(span.get("page_no", -1)) <= 36:
                if "total assets" in (span.get("text") or "").lower() or "assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
