def rule_item1_business_trades_on_exchange(doc: dict) -> list[dict]:
    """Match Item 1 / Business spans saying the stock trades on an exchange."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = span.get("text", "") or ""
            if re.search(r"item\s*1|business", path, re.I) and re.search(r"trades on .*?(nasdaq|stock exchange|nyse)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
