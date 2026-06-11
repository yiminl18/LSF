def rule_table_registered_exchange_header(doc: dict) -> list[dict]:
    """Match tables whose text contains 'registered' and 'exchange' in the header area."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "exchange" in txt and "registered" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
