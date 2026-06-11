def rule_page1_two_spans_before_address_label(doc: dict) -> list[dict]:
    """Match the two spans immediately preceding an address label, useful when address is split across lines."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1:
                txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if re.search(r'address of principal executive offices', txt, re.I):
                    for k in (i - 2, i - 1):
                        if 0 <= k < len(texts) and texts[k].get("page_no") == 1:
                            out.append(texts[k])
        return out
    except Exception:
        return []
