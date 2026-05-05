def rule_tables_with_consolidated_keyword(doc: dict) -> list[dict]:
    """Match tables with 'consolidated' plus balance-sheet-like content."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "consolidated" in text and ("assets" in text or "balance sheet" in text or "financial position" in text):
                out.append(span)
        return out
    except Exception:
        return []
