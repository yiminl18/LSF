def rule_item1_business_overview_address_sentence(doc: dict) -> list[dict]:
    """Match Business overview spans that include incorporation history and office address."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "item 1" in path and "business" in path:
                if "reincorporated in delaware" in txt and ("executive offices" in txt or "principal facilities" in txt):
                    out.append(span)
        return out
    except Exception:
        return []
