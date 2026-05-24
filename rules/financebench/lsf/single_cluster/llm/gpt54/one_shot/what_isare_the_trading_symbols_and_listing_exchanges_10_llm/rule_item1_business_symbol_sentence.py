def rule_item1_business_symbol_sentence(doc: dict) -> list[dict]:
    """Match Item 1 / Business spans that explicitly state stock trades under a symbol."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = ((span.get("structure", {}) or {}).get("path_text") or "").lower()
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "item 1" in path and "business" in path:
                if "under the symbol" in txt or "trades on" in txt or "listed on the" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
