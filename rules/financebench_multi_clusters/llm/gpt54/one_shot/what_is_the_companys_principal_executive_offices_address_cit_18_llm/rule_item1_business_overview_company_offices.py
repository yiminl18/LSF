def rule_item1_business_overview_company_offices(doc: dict) -> list[dict]:
    """Match Item 1 Business overview spans mentioning company offices in city/state."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "item 1" in path and "business" in path:
                if "our principal corporate offices are located in" in txt or "our executive offices and principal facilities are located at" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
