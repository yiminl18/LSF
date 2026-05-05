def rule_tables_with_total_assets_and_total_liabilities_equity(doc: dict) -> list[dict]:
    """Match tables containing total assets and total liabilities/equity balancing lines."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and (
                "total liabilities and equity" in low
                or "total liabilities and shareholders' equity" in low
                or "total liabilities and stockholders' equity" in low
                or "total liabilities and shareowners' equity" in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
