def rule_table_with_exchange_column(doc: dict) -> list[dict]:
    """Match table spans whose cells include an exchange-registration column header."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if any(
                re.search(r"name of each exchange on which registered", (c.get("text", "") or ""), re.I)
                or re.search(r"name of each exchange", (c.get("text", "") or ""), re.I)
                or re.search(r"name of exchange on which registered", (c.get("text", "") or ""), re.I)
                for c in cells
            ):
                out.append(span)
        return out
    except Exception:
        return []
