def rule_page1_body_after_h2_h3_listing_headers(doc: dict) -> list[dict]:
    """Return body spans immediately after page-1 H2/H3 listing headers."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            lvl = ((span.get("structure") or {}).get("level") or "")
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and lvl in {"H2", "H3"} and (
                re.search(r"trading symbol", txt, re.I)
                or re.search(r"name of each exchange", txt, re.I)
            ):
                for j in range(i + 1, min(i + 5, len(texts))):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
