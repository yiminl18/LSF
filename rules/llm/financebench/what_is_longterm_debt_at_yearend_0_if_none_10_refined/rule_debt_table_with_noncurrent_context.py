def rule_debt_table_with_noncurrent_context(doc: dict) -> list[dict]:
    """Match tables where long-term debt appears in non-current liabilities context."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if (
                re.search(r"\blong[\-\s]?term debt\b", txt)
                and ("non-current" in txt or "noncurrent" in txt or "long-term liabilities" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
