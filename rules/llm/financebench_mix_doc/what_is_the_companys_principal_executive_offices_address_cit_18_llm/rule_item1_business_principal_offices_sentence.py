def rule_item1_business_principal_offices_sentence(doc: dict) -> list[dict]:
    """Match Item 1 / Business text spans mentioning principal or executive offices are located."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "item 1" in path and "business" in path:
                if "principal corporate offices are located" in txt or "executive offices and principal facilities are located" in txt or "executive offices are located at" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
