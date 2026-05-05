def rule_page1_after_ein_before_phone(doc: dict) -> list[dict]:
    """Match spans on page 1 that look like address lines positioned between EIN and phone labels."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if not txt:
                continue
            if not re.search(r'^\d{1,5}\s', txt):
                continue
            window = " ".join(
                ((texts[j].get("text") or "") + " " + (texts[j].get("text_span") or ""))
                for j in range(max(0, i-3), min(len(texts), i+4))
                if texts[j].get("page_no") == 1
            )
            if re.search(r'employer identification|i\.r\.s\.', window, re.I) and re.search(r'telephone number|area code|\(\d{3}\)', window, re.I):
                out.append(span)
        return out
    except Exception:
        return []
