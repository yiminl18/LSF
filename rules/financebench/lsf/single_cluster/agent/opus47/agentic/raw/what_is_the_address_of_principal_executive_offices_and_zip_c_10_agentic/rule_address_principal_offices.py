def rule_address_principal_offices(doc: dict) -> list[dict]:
    '''Cover-page address: spans within a small window around "(Address of principal executive offices)" or "(Zip Code)" labels on page 1.'''
    texts = doc.get("texts", [])
    keep = set()
    n = len(texts)
    for i, span in enumerate(texts):
        if span.get("page_no") != 1:
            continue
        t = (span.get("text") or "").lower()
        is_addr_label = ("address" in t) and ("principal executive offices" in t)
        is_zip_label = "(zip code)" in t
        if is_addr_label or is_zip_label:
            lo = max(0, i - 5)
            hi = min(n, i + 2)
            for j in range(lo, hi):
                if texts[j].get("page_no") == 1:
                    keep.add(j)
    return [texts[j] for j in sorted(keep)]
