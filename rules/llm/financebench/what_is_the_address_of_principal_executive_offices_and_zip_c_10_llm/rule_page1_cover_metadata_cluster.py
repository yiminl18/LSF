def rule_page1_cover_metadata_cluster(doc: dict) -> list[dict]:
    """Match the dense cover-page metadata cluster around state, EIN, address, ZIP, and phone."""
    try:
        spans = doc.get("texts", [])
        trigger_idxs = []
        for i, span in enumerate(spans):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if (
                "state or other jurisdiction of incorporation" in txt
                or "employer identification no" in txt
                or "zip code" in txt
                or "telephone number" in txt
            ):
                trigger_idxs.append(i)
        out_idx = set()
        for i in trigger_idxs:
            for j in range(max(0, i - 3), min(len(spans), i + 3)):
                if spans[j].get("page_no") == 1:
                    out_idx.add(j)
        return [spans[i] for i in sorted(out_idx)]
    except Exception:
        return []
