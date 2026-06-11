def rule_exchange_near_cover_phone_and_address_block(doc: dict) -> list[dict]:
    """Match exchange spans in the dense cover-page registrant information block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "")
            txt = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if span.get("page_no") == 1 and path and re.search(r'new york stock exchange|nasdaq|global select market', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
