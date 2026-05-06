def rule_page1_table_or_text_with_title_symbol_exchange_triplet(doc: dict) -> list[dict]:
    """Match spans containing the triplet of title/class, symbol, and exchange cues."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if (
                "title of each class" in txt
                and ("trading symbol" in txt or "trading symbol(s)" in txt)
                and "exchange" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
