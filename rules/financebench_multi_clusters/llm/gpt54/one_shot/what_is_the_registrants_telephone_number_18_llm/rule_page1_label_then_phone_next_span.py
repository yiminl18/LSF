def rule_page1_label_then_phone_next_span(doc: dict) -> list[dict]:
    """Match a page-1 telephone-label span followed by a standalone phone span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"^\s*(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}\s*$")
        for i, span in enumerate(texts[:-1]):
            if span.get("page_no") != 1:
                continue
            if re.search(r"telephone number|area code", span.get("text", "") or "", re.I):
                nxt = texts[i + 1]
                if nxt.get("page_no") == 1 and phone_re.match((nxt.get("text", "") or "").strip()):
                    out.append(span)
                    out.append(nxt)
        return out
    except Exception:
        return []
