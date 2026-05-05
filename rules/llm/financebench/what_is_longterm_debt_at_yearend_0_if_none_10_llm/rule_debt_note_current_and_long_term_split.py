def rule_debt_note_current_and_long_term_split(doc: dict) -> list[dict]:
    """Match debt note tables that split current and long-term portions of debt."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if (
                ("current portion" in txt or "current maturities" in txt)
                and ("long-term debt" in txt or "long term debt" in txt or "debt excluding current maturities" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
