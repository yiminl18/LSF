def rule_table_with_debt_and_total_liabilities(doc: dict) -> list[dict]:
    """Match liability tables that contain long-term debt together with total liabilities or total debt context."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if re.search(r"\blong[\-\s]?term debt\b", txt) and (
                "total liabilities" in txt
                or "total debt" in txt
                or "liabilities" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
