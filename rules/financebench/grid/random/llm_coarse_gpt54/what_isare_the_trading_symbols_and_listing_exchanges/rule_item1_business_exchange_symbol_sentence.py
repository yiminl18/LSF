def rule_item1_business_exchange_symbol_sentence(doc: dict) -> list[dict]:
    """Match Item 1 / Business spans mentioning both an exchange and a ticker symbol in one sentence."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"item\s*1|business", path, re.I) and re.search(r"symbol", txt, re.I) and re.search(r"nasdaq|stock exchange|nyse", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
