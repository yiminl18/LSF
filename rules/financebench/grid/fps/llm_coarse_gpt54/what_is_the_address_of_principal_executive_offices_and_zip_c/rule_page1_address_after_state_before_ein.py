def rule_page1_address_after_state_before_ein(doc: dict) -> list[dict]:
    """Match address-like spans that appear between state-of-incorporation and EIN spans on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        state_idx = None
        ein_idx = None
        for i, s in enumerate(texts):
            if s.get("page_no") != 1:
                continue
            full = (s.get("text") or "") + " " + (s.get("text_span") or "")
            if state_idx is None and re.search(r'state or other jurisdiction of incorporation|jurisdiction of incorporation', full, re.I):
                state_idx = i
            if ein_idx is None and re.search(r'employer identification no|i\.r\.s\. employer identification', full, re.I):
                ein_idx = i
        if state_idx is None or ein_idx is None or ein_idx <= state_idx:
            return []
        for s in texts[state_idx:ein_idx+1]:
            t = (s.get("text") or "").strip()
            if re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I):
                out.append(s)
        return out
    except Exception:
        return []
