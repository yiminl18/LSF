def rule_tables_with_total_assets_and_receivables(doc: dict) -> list[dict]:
    """Match balance-sheet-like tables containing total assets and receivables."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and ("receivables" in low or "accounts receivable" in low):
                out.append(span)
        return out
    except Exception:
        return []
