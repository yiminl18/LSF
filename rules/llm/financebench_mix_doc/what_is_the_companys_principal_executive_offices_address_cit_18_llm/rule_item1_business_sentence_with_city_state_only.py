def rule_item1_business_sentence_with_city_state_only(doc: dict) -> list[dict]:
    """Match Item 1 Business text spans that mention only the city/state office location sentence."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "item 1" in path and "business" in path:
                if "located in seattle, washington" in txt or "located in san jose, california" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
