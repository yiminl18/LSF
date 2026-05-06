def rule_ebay_overview_incorporation(doc: dict) -> list[dict]:
    """Retrieve the eBay overview paragraph containing the incorporation sentence."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 4:
                continue
            if span.get("label") != "text":
                continue
            structure = span.get("structure") or {}
            path = (structure.get("path_text") or "")
            txt = (span.get("text") or "")
            if "ITEM 1: BUSINESS | Overview" not in path:
                continue
            low = txt.lower()
            if "was formed as a sole proprietorship" in low and "reincorporated in delaware" in low:
                out.append(span)
        return out
    except Exception:
        return []

