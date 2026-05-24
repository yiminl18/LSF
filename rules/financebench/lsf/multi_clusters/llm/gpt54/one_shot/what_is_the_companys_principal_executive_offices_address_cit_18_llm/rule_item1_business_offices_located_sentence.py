def rule_item1_business_offices_located_sentence(doc: dict) -> list[dict]:
    """Match Item 1 / Business text spans with 'offices are located at' phrasing."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "item 1" in path and "business" in path and "located at" in txt:
                if "executive offices" in txt or "principal facilities" in txt or "principal corporate offices" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
