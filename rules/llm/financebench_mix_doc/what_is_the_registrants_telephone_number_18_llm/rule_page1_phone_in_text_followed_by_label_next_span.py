def rule_page1_phone_in_text_followed_by_label_next_span(doc: dict) -> list[dict]:
    """Match a page-1 phone-only span followed by a label span like '(Registrant’s telephone number...)'."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"^\s*(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}\s*$")
        for i, span in enumerate(texts[:-1]):
            if span.get("page_no") != 1:
                continue
            if phone_re.match((span.get("text", "") or "").strip()):
                nxt = texts[i + 1]
                if nxt.get("page_no") == 1 and re.search(r"telephone number|area code", nxt.get("text", "") or "", re.I):
                    out.append(span)
                    out.append(nxt)
        return out
    except Exception:
        return []
