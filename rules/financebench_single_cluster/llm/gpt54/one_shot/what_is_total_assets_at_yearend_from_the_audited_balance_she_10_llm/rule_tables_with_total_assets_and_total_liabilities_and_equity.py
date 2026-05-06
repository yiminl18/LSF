def rule_tables_with_total_assets_and_total_liabilities_and_equity(doc: dict) -> list[dict]:
    """Match tables containing total assets and a combined total liabilities and equity line."""
    try:
        out = []
        phrases = [
            "total liabilities and equity",
            "total liabilities and shareholders' equity",
            "total liabilities and stockholders' equity",
            "total liabilities and shareowners' equity",
        ]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and any(p in low for p in phrases):
                out.append(span)
        return out
    except Exception:
        return []
