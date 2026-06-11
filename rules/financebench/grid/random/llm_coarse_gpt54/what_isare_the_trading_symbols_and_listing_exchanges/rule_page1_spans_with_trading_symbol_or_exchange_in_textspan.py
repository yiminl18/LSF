def rule_page1_spans_with_trading_symbol_or_exchange_in_textspan(doc: dict) -> list[dict]:
    """Match page-1 spans whose text_span contains trading symbol or exchange wording."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text_span", "") or ""
            if span.get("page_no") == 1 and (
                re.search(r"trading symbol", txt, re.I)
                or re.search(r"name of each exchange", txt, re.I)
                or re.search(r"new york stock exchange|nasdaq", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
