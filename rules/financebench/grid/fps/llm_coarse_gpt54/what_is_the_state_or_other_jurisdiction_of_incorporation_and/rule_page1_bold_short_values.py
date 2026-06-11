def rule_page1_bold_short_values(doc: dict) -> list[dict]:
    """Match short bold page-1 spans that are likely the state or EIN values."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if len(txt) <= 20 and (
                re.fullmatch(r"\d{2}-\d{7}", txt) or
                re.fullmatch(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
