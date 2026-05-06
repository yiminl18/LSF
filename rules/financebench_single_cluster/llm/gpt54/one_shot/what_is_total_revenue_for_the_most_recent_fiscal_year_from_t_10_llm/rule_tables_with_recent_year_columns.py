def rule_tables_with_recent_year_columns(doc: dict) -> list[dict]:
    """Match financial tables that contain recent year columns and a revenue row."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            texts = [(c.get("text") or "") for c in cells]
            joined = " ".join(texts).lower()
            if not any(k in joined for k in ["revenue", "revenues", "sales", "net sales", "net revenues"]):
                continue
            years = set()
            for t in texts:
                for y in re.findall(r"\b(20\d{2}|19\d{2})\b", t):
                    years.add(y)
            if len(years) >= 2:
                out.append(span)
        return out
    except Exception:
        return []
