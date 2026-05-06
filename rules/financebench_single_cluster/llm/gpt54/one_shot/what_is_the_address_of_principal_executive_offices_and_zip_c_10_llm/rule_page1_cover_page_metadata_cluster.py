def rule_page1_cover_page_metadata_cluster(doc: dict) -> list[dict]:
    """Return the cluster of page-1 spans around state/EIN/address/ZIP metadata."""
    try:
        spans = doc.get("texts", [])
        idxs = []
        for i, s in enumerate(spans):
            if s.get("page_no") != 1:
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
            if (
                "state or other jurisdiction" in txt
                or "state of incorporation" in txt
                or "i.r.s. employer identification" in txt
                or "irs employer identification" in txt
                or "(zip code)" in txt
                or "address of principal executive offices" in txt
            ):
                idxs.append(i)
        out_idx = set()
        for i in idxs:
            for j in range(max(0, i - 1), min(len(spans), i + 2)):
                if spans[j].get("page_no") == 1:
                    out_idx.add(j)
        return [spans[i] for i in sorted(out_idx)]
    except Exception:
        return []
