def rule_tables_with_total_assets_and_not_cover_page(doc: dict) -> list[dict]:
    """Match total-assets tables not on the cover page."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and int(span.get("page_no", 0)) > 1:
                if "total assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
