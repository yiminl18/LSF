def rule_ebay_item1_overview_body(doc: dict) -> list[dict]:
    """Retrieve the company-identifying overview paragraph under Item 1 Business for eBay-style filings."""
    try:
        spans = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 4:
                continue
            if span.get("label") != "text":
                continue
            structure = span.get("structure") or {}
            path = (structure.get("path_text") or "")
            if path != "Form 10-K | PART I | ITEM 1: BUSINESS | Overview":
                continue
            text = (span.get("text") or "")
            low = text.lower()
            if "was formed as a sole proprietorship" in low or "reincorporated in delaware" in low:
                spans.append(span)
        return spans
    except Exception:
        return []

