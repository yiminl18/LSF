def rule_item1_business_stock_listed(doc: dict) -> list[dict]:
    """Match Item 1 / Business spans stating the common stock is listed or trades under a symbol."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"item\s*1|business", path, re.I) and (
                re.search(r"common stock .* listed .* under the symbol", txt, re.I)
                or re.search(r"common stock trades .* under the symbol", txt, re.I)
                or re.search(r"trades on .* under the symbol", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
