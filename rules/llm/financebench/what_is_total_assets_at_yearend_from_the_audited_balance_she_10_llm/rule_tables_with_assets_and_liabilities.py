def rule_tables_with_assets_and_liabilities(doc: dict) -> list[dict]:
    """Match balance-sheet-like tables containing both assets and liabilities/equity language."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            has_assets = "assets" in text
            has_liab = "liabilities" in text or "shareholders' equity" in text or "stockholders' equity" in text or "equity" in text
            if has_assets and has_liab:
                out.append(span)
        return out
    except Exception:
        return []
