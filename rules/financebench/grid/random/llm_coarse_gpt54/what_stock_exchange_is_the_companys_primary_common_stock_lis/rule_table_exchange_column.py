def rule_table_exchange_column(doc: dict) -> list[dict]:
    """Match table spans containing a column header for exchange registration."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") == "table":
                cells = (((span.get("table_data") or {}).get("cells")) or [])
                joined = " ".join((c.get("text") or "") for c in cells).lower()
                if "name of each exchange on which registered" in joined or "name of each exchange on which registered" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
