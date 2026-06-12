def rule_tables_with_fiscal_year_rows(doc: dict) -> list[dict]:
    """Match tables that include fiscal-year rows and debt/public columns, useful for year-end answer extraction."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if (
                "fiscal year" in txt
                and ("held by the public" in txt or "debt held by the public" in txt or "summary of federal debt" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
