def rule_page1_after_ein_before_phone(doc: dict) -> list[dict]:
    """Match spans on page 1 positioned between EIN and telephone labels in the registrant block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'I\.?R\.?S\.? Employer Identification No', txt, re.I):
                for j in range(i + 1, min(i + 8, len(texts))):
                    s2 = texts[j]
                    if s2.get("page_no") != 1:
                        break
                    t2 = (s2.get("text") or "") + " " + (s2.get("text_span") or "")
                    if re.search(r'telephone number', t2, re.I):
                        break
                    out.append(s2)
        return out
    except Exception:
        return []
