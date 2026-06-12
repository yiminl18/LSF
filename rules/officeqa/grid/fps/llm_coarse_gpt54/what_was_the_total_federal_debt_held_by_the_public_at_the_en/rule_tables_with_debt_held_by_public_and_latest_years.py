def rule_tables_with_debt_held_by_public_and_latest_years(doc: dict) -> list[dict]:
    """Match debt-held-by-public tables that include multiple annual rows, indicating year-end totals."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            year_hits = sum(1 for y in ["1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1997", "2008", "2015", "2016", "2017", "2018", "2024"] if y in txt)
            if ("debt held by the public" in txt or "held by the public" in txt) and year_hits >= 2:
                out.append(span)
        return out
    except Exception:
        return []
