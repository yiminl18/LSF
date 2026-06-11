def rule_exchange_after_item_9_exhibits_for_8k_cover(doc: dict) -> list[dict]:
    """High-recall fallback: match exchange mentions anywhere in 8-K cover metadata on page 1."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'securities registered pursuant to section 12\(b\)', txt, re.I) or re.search(r'new york stock exchange|nasdaq|global select market', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
