def rule_page1_short_bold_text_after_exact_name(doc: dict) -> list[dict]:
    """Match short bold text spans after the exact-name label, often the state and EIN values."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        start = None
        for i, span in enumerate(texts):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r"exact name of registrant", txt, re.I):
                start = i
                break
        if start is None:
            return []
        for span in texts[start:start+10]:
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if len(txt) <= 25 and (
                re.fullmatch(r"\d{2}-\d{7}", txt) or
                re.fullmatch(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
