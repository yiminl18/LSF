def rule_tables_with_total_assets_on_pages_40_to_120(doc: dict) -> list[dict]:
    """Match total-assets tables on later financial-statement page range 40-120."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and 40 <= int(span.get("page_no", 0)) <= 120:
                if "total assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
