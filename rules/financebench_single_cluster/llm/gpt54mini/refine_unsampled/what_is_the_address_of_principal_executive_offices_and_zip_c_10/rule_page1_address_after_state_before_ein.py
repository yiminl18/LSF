def rule_page1_address_after_state_before_ein(doc: dict) -> list[dict]:
    """Match spans between state-of-incorporation and EIN labels, where address often sits."""
    try:
        spans = doc.get("texts", [])
        state_i = None
        ein_i = None
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if state_i is None and ("state or other jurisdiction of incorporation" in txt or "state of incorporation" in txt):
                state_i = i
            if ein_i is None and "employer identification no" in txt:
                ein_i = i
        if state_i is None or ein_i is None or ein_i <= state_i:
            return []
        return [spans[i] for i in range(state_i, ein_i + 1) if spans[i].get("page_no") == 1]
    except Exception:
        return []
