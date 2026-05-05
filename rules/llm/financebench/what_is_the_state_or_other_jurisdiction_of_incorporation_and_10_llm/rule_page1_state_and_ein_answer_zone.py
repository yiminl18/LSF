def rule_page1_state_and_ein_answer_zone(doc: dict) -> list[dict]:
    """Match the compact answer zone on page 1 where state and EIN usually appear before the address block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        start = None
        end = None
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if start is None and span.get("page_no") == 1 and "exact name of registrant" in txt:
                start = i
            if start is not None and end is None and span.get("page_no") == 1 and re.search(r"address of principal executive offices|registrant.?s telephone number", txt, re.I):
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 15)
        return [s for s in texts[start:end] if s.get("page_no") == 1]
    except Exception:
        return []
