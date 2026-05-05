def rule_income_statement_nearby_tables(doc: dict) -> list[dict]:
    """Match tables appearing shortly after an income-statement-like section header."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if not any(k in txt for k in [
                "statement of income", "statements of income",
                "statement of operations", "statements of operations",
                "statement of earnings", "statements of earnings",
                "income statement"
            ]):
                continue
            for j in range(i + 1, min(i + 6, len(texts))):
                nxt = texts[j]
                if nxt.get("label") == "table":
                    out.append(nxt)
        return out
    except Exception:
        return []
