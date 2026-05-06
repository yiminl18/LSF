def rule_tables_with_total_assets_on_late_pages(doc: dict) -> list[dict]:
    """Match total-assets tables on later pages, where audited statements usually appear rather than cover pages."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and int(span.get("page_no", 0)) >= 20:
                if "total assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
