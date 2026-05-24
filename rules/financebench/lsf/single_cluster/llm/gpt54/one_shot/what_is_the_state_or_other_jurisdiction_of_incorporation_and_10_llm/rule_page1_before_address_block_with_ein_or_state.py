def rule_page1_before_address_block_with_ein_or_state(doc: dict) -> list[dict]:
    """Match page-1 spans before the address block that contain state/incorporation/EIN cues."""
    try:
        import re
        texts = doc.get("texts", [])
        end = None
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and re.search(r"address of principal executive offices", txt, re.I):
                end = i
                break
        if end is None:
            end = len(texts)
        out = []
        for span in texts[:end]:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"\d{2}-\d{7}|state|jurisdiction|incorporation|employer identification", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
