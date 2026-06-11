def rule_table_exchange_column(doc: dict) -> list[dict]:
    """Match table spans whose cells include an exchange-name column/value."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            joined = " | ".join((c.get("text") or "") for c in cells)
            if re.search(r'name of each exchange on which registered', joined, re.I) and re.search(
                r'new york stock exchange|nasdaq|global select market', joined, re.I
            ):
                out.append(span)
        return out
    except Exception:
        return []
