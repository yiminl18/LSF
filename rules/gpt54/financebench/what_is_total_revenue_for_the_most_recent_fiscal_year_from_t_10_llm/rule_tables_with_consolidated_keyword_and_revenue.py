def rule_tables_with_consolidated_keyword_and_revenue(doc: dict) -> list[dict]:
    """Match tables mentioning consolidated plus revenue/sales keywords."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = ((span.get("text") or "") + " " + ((span.get("structure", {}) or {}).get("path_text", "") or "")).lower()
            if "consolidated" in txt and any(k in txt for k in ["revenue", "revenues", "sales", "net sales", "net revenues"]):
                out.append(span)
        return out
    except Exception:
        return []
