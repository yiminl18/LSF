def rule_business_section_listed_on_exchange(doc: dict) -> list[dict]:
    """Match Item 1 / Business text spans that say the common stock is listed or trades on an exchange."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "item 1" in path or "business" in path:
                if ("listed on" in txt or "trades on" in txt or "primary listing on" in txt) and (
                    "stock exchange" in txt or "nasdaq" in txt or "nyse" in txt
                ):
                    out.append(span)
                elif re.search(r"\b(nasdaq|new york stock exchange|nasdaq global select market)\b", txt):
                    out.append(span)
        return out
    except Exception:
        return []
