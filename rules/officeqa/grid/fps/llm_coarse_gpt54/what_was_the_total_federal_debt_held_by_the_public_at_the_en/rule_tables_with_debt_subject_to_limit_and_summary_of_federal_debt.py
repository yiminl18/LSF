def rule_tables_with_debt_subject_to_limit_and_summary_of_federal_debt(doc: dict) -> list[dict]:
    """Match Federal Debt summary tables that list neighboring FD tables like debt subject to limit, indicating the FD section TOC/summary area."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "summary of federal debt" in txt and "debt subject to" in txt:
                out.append(span)
        return out
    except Exception:
        return []
