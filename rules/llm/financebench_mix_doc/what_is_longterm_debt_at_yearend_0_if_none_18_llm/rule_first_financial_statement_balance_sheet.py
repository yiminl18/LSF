def rule_first_financial_statement_balance_sheet(doc: dict) -> list[dict]:
    """Match the first balance sheet table in the document."""
    out = []
    try:
        candidates = []
        for i, span in enumerate(doc.get("texts", [])):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "balance sheet" in txt or "balance sheets" in txt:
                    candidates.append((i, span))
        if candidates:
            out.append(sorted(candidates, key=lambda x: x[0])[0][1])
    except Exception:
        return []
    return out
