def rule_tables_with_total_assets_and_total_liabilities(doc: dict) -> list[dict]:
    """Match tables containing both total assets and total liabilities."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "total assets" in text and "total liabilities" in text:
                out.append(span)
        return out
    except Exception:
        return []
