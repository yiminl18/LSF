def rule_item1_business_listed_under_symbol(doc: dict) -> list[dict]:
    """Match Item 1 / Business spans saying the stock is listed under a symbol."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = span.get("text", "") or ""
            if re.search(r"item\s*1|business", path, re.I) and re.search(r"listed .* under the symbol", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
