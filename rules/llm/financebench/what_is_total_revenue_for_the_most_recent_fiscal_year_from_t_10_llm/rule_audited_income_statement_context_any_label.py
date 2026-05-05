def rule_audited_income_statement_context_any_label(doc: dict) -> list[dict]:
    """Match any span in audited income-statement context, not just tables, for high recall."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure", {}) or {}).get("path_text", "") or "").lower()
            txt = (span.get("text") or "").lower()
            hay = path + " " + txt
            if any(k in hay for k in [
                "statement of income", "statements of income",
                "statement of operations", "statements of operations",
                "statement of earnings", "statements of earnings",
                "financial statements", "supplementary data", "item 8"
            ]):
                out.append(span)
        return out
    except Exception:
        return []
