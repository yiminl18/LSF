def rule_jnj_page1_trading_symbols_exchange(doc: dict) -> list[dict]:
    """Retrieve Johnson & Johnson page-1 trading symbol / exchange cluster."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            low = f"{path} {txt}".lower()
            if span.get("label") in {"text", "section_header", "table"} and (
                "johnson & johnson" in low or "trading symbol" in low or "new york stock exchange" in low
            ):
                out.append(span)
        return out
    except Exception:
        return []

