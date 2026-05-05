def rule_page1_address_near_state_and_ein(doc: dict) -> list[dict]:
    """Return spans between state-of-incorporation and EIN labels, which often include the address."""
    try:
        spans = doc.get("texts", [])
        state_idx = ein_idx = None
        for i, s in enumerate(spans):
            if s.get("page_no") != 1:
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
            if state_idx is None and ("state or other jurisdiction" in txt or "state of incorporation" in txt):
                state_idx = i
            if ein_idx is None and ("i.r.s. employer identification" in txt or "irs employer identification" in txt):
                ein_idx = i
        if state_idx is None or ein_idx is None:
            return []
        lo, hi = sorted([state_idx, ein_idx])
        return [s for s in spans[max(0, lo - 2):min(len(spans), hi + 2)] if s.get("page_no") == 1]
    except Exception:
        return []
