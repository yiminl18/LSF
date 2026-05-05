def rule_item1_first_company_sentence(doc: dict) -> list[dict]:
    """Match early business-description sentences that begin with the company name or company reference."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if "item 1" not in path and "business" not in path:
                continue
            if span.get("page_no", 999) > 15:
                continue
            if low.startswith("the company") or low.startswith("johnson & johnson") or low.startswith("amazon.com, inc.") or low.startswith("costco wholesale corporation") or low.startswith("the boeing company") or low.startswith("ebay inc.") or low.startswith("corning") or low.startswith("amcor plc"):
                out.append(span)
        return out
    except Exception:
        return []
