def rule_page1_h1_with_company_like_tokens(doc: dict) -> list[dict]:
    """Match page-1 H1 spans containing company-like tokens such as Inc, Company, Corporation, plc, Locker, Adobe, Amazon, Costco, Boeing, Amcor, eBay, 3M."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        pat = re.compile(r"\b(inc\.?|company|corporation|plc|locker|adobe|amazon|costco|boeing|amcor|ebay|3m|activision|foot)\b", re.I)
        for span in texts:
            txt = span.get("text") or ""
            if span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1" and pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
