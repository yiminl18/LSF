def rule_tables_with_public_debt_and_latest_balance_columns(doc: dict) -> list[dict]:
    """Match tables that look like balance tables for public debt with end-of-period columns."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if (
                ("end of period" in txt or "end of quarter" in txt or "end of fiscal year" in txt)
                and ("public debt" in txt or "held by the public" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
