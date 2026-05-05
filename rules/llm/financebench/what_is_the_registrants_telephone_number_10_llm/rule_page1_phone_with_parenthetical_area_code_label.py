def rule_page1_phone_with_parenthetical_area_code_label(doc: dict) -> list[dict]:
    """Match phone spans adjacent to '(Registrant's telephone number, including area code)' labels."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "")
            if not phone_re.search(text):
                continue
            neigh = texts[max(0, i-3): min(len(texts), i+4)]
            neigh_text = " ".join((n.get("text") or "") for n in neigh)
            if re.search(r"registrant[’'`s]{0,2}\s+telephone\s+number,\s*including\s+area\s+code", neigh_text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
