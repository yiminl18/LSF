def rule_page1_h2_h3_listing_headers(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 headers around the listing block, such as 'Trading Symbol(s)' or exchange-name headers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            lvl = ((span.get("structure") or {}).get("level") or "")
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and lvl in {"H2", "H3"} and (
                re.search(r"trading symbol", txt, re.I)
                or re.search(r"name of each exchange", txt, re.I)
                or re.search(r"name of exchange on which registered", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
