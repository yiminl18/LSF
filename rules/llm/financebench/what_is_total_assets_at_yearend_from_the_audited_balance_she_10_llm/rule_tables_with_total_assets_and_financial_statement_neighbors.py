def rule_tables_with_total_assets_and_financial_statement_neighbors(doc: dict) -> list[dict]:
    """Match total-assets tables with neighboring spans mentioning other primary financial statements."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            if "total assets" not in (span.get("text") or "").lower():
                continue
            neigh = texts[max(0, i-10):min(len(texts), i+10)]
            joined = " ".join((n.get("text") or "").lower() for n in neigh)
            if any(k in joined for k in ["cash flows", "income", "comprehensive income", "equity", "notes to"]):
                out.append(span)
        return out
    except Exception:
        return []
