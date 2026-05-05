def rule_page1_address_or_zip_neighbors(doc: dict) -> list[dict]:
    """Return spans adjacent to page-1 address/ZIP label spans."""
    try:
        spans = doc.get("texts", [])
        idxs = set()
        for i, span in enumerate(spans):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if (
                "address of principal executive offices" in txt
                or "address and telephone number, including area code, of registrant" in txt
                or "address of principal executive offices and zip code" in txt
                or "(zip code)" in txt
            ):
                for j in range(max(0, i - 2), min(len(spans), i + 3)):
                    if spans[j].get("page_no") == 1:
                        idxs.add(j)
        return [spans[i] for i in sorted(idxs)]
    except Exception:
        return []
