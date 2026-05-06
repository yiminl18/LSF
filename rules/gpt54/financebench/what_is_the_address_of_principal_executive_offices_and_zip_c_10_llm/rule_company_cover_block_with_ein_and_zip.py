def rule_company_cover_block_with_ein_and_zip(doc: dict) -> list[dict]:
    """Match spans in the cover block near both EIN and ZIP references."""
    try:
        spans = doc.get("texts", [])
        out_idx = set()
        trigger_idxs = []
        for i, span in enumerate(spans):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "i.r.s. employer identification" in txt or "irs employer identification" in txt or "(zip code)" in txt:
                trigger_idxs.append(i)
        for i in trigger_idxs:
            for j in range(max(0, i - 3), min(len(spans), i + 3)):
                if spans[j].get("page_no") == 1:
                    out_idx.add(j)
        return [spans[i] for i in sorted(out_idx)]
    except Exception:
        return []
