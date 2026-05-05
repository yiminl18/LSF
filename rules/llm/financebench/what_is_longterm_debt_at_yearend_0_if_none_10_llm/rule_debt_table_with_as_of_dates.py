def rule_debt_table_with_as_of_dates(doc: dict) -> list[dict]:
    """Match tables with debt rows and as-of date headers like December 31 or June 30."""
    import re
    try:
        out = []
        date_re = re.compile(r"(december|june|may|january|august|september)\s+\d{1,2}?,?\s+\d{4}", re.I)
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            low = txt.lower()
            if (re.search(r"\blong[\-\s]?term debt\b", low) or "debt excluding current maturities" in low) and (
                "as of" in low or date_re.search(txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
