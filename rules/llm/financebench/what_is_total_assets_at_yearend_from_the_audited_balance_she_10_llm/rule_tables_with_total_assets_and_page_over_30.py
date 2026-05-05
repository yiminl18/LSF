def rule_tables_with_total_assets_and_page_over_30(doc: dict) -> list[dict]:
    """Match total-assets tables on later pages, where audited statements often reside in 10-Ks."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and int(span.get("page_no", 0)) >= 30:
                if "total assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
