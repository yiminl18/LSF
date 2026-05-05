def rule_jnj_multi_symbol_exchange_table(doc: dict) -> list[dict]:
    """Match Johnson & Johnson-style page 1 registration table rows with multiple trading symbols and same exchange."""
    try:
        import re
        out = []
        texts = doc.get("texts", [])
        for span in texts:
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r"trading symbol|name of each exchange on which registered", txt, re.I):
                out.append(span)
        for span in texts:
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.fullmatch(r"JNJ(?:\d+[A-Z]{0,3})?", txt.strip(), re.I):
                out.append(span)
            if span.get("page_no") == 1 and re.search(r"new york stock exchange", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
