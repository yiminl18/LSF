def rule_tables_with_total_assets_and_two_or_more_asset_lines(doc: dict) -> list[dict]:
    """Match tables with total assets and multiple asset-related lines."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            asset_terms = sum(1 for term in ["current assets", "other assets", "total assets", "assets"] if term in txt)
            if "total assets" in txt and asset_terms >= 2:
                out.append(span)
        return out
    except Exception:
        return []
