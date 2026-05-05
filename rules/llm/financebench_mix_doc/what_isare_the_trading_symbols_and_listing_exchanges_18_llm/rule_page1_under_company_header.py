def rule_page1_under_company_header(doc: dict) -> list[dict]:
    """Match body spans on page 1 under the main company H1 that include listing-related content."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "form 10-k" not in path and (
                "adobe inc." in path
                or "amazon.com, inc." in path
                or "costco wholesale corporation" in path
                or "the boeing company" in path
                or "activision blizzard, inc." in path
                or "amcor plc" in path
                or "foot locker, inc." in path
                or "ebay" in path
                or "3m company" in path
            ):
                if (
                    "trading symbol" in txt
                    or "exchange on which registered" in txt
                    or "nasdaq" in txt
                    or "stock exchange" in txt
                ):
                    out.append(span)
        return out
    except Exception:
        return []
