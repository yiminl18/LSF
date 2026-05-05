def rule_tables_with_assets_liabilities_equity_triplet(doc: dict) -> list[dict]:
    """Match tables containing assets, liabilities, and equity together."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "assets" in text and "liabilities" in text and "equity" in text:
                out.append(span)
        return out
    except Exception:
        return []
