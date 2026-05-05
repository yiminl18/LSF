def rule_jnj_multi_symbol_exchange_cover(doc: dict) -> list[dict]:
    """Match Johnson & Johnson-style cover-page spans where multiple trading symbols/exchanges may be listed."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and (
                "trading symbol" in txt
                or "securities registered pursuant to section 12(b)" in txt
                or "new york stock exchange" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
