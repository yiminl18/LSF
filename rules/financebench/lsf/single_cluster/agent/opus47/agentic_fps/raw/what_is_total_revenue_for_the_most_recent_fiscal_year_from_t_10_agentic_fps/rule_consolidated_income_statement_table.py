def rule_consolidated_income_statement_table(doc: dict) -> list[dict]:
    """Tables under the audited Consolidated Statements of Operations/Earnings/Income section."""
    keywords = (
        "consolidated statements of operations",
        "consolidated statements of earnings",
        "consolidated statements of income",
        "consolidated statement of operations",
        "consolidated statement of earnings",
        "consolidated statement of income",
    )
    out = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        path = (span.get("structure") or {}).get("path_text", "") or ""
        text = span.get("text", "") or ""
        path_l = path.lower()
        text_l = text.lower()
        if any(k in path_l for k in keywords) or any(k in text_l for k in keywords):
            out.append(span)
    return out
