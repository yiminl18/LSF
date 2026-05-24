def rule_page_range_financial_statement_tables(doc: dict) -> list[dict]:
    """Match tables on later pages where audited financial statements usually appear."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and (span.get("page_no") or 0) >= 30:
                txt = (span.get("text") or "").lower()
                path = (span.get("structure", {}) or {}).get("path_text", "").lower()
                if any(k in (txt + " " + path) for k in ["revenue", "revenues", "sales", "income", "operations", "earnings"]):
                    out.append(span)
        return out
    except Exception:
        return []
