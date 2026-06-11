def rule_exchange_in_cover_parenthetical_block(doc: dict) -> list[dict]:
    """Match exchange mentions in the same cover block that contains charter/address/phone metadata."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if span.get("page_no") == 1 and re.search(r'exact name of registrant|address of principal executive offices|telephone number', combined, re.I):
                continue
            if span.get("page_no") == 1 and re.search(r'new york stock exchange|nasdaq|global select market', combined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
