def rule_company_header_with_trading_symbol_embedded(doc: dict) -> list[dict]:
    """Match page-1 company header spans whose text or text_span contains trading symbol language."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            blob = ((span.get("text", "") or "") + " " + (span.get("text_span", "") or "")).strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header" and re.search(r"trading symbol", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
