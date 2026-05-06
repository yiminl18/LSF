def rule_tables_with_total_assets_on_pages_30_to_80(doc: dict) -> list[dict]:
    """Match total-assets tables on common financial-statement page range 30-80."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and 30 <= int(span.get("page_no", 0)) <= 80:
                if "total assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
