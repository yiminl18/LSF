def rule_debt_note_year_end_table(doc: dict) -> list[dict]:
    """Match debt tables that explicitly present year-end balances."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if ("year ended" in txt or "as of" in txt or "at december" in txt or "at june" in txt) and (
                re.search(r"\blong[\-\s]?term debt\b", txt) or "borrowings" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
