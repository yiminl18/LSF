def rule_page1_ebay_sanjose_overview_sentence(doc: dict) -> list[dict]:
    """Match eBay-style overview sentence stating principal executive offices are located at an address."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'Overview|ITEM 1: BUSINESS', path, re.I) and re.search(r'principal executive offices are located at .*San Jose,\s*California', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
