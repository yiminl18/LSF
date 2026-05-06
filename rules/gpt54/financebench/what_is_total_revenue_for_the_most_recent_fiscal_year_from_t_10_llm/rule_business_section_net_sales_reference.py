def rule_business_section_net_sales_reference(doc: dict) -> list[dict]:
    """Match business-section spans that explicitly refer to net sales being in Item 8/segment note."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure", {}) or {}).get("path_text", "") or "").lower()
            txt = (span.get("text") or "").lower()
            if "item 1" in path and "business" in path and "net sales" in txt and ("item 8" in txt or "financial statements" in txt):
                out.append(span)
        return out
    except Exception:
        return []
