def rule_page1_bold_smallfont_registration_values(doc: dict) -> list[dict]:
    """Match page 1 small-font bold spans likely used for ticker or exchange values in top matter."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            size = span.get("size") or 0
            if span.get("page_no") != 1:
                continue
            if size <= 9.5 and span.get("bold") == 1:
                if re.fullmatch(r"[A-Z]{1,8}(?:\d+[A-Z]{0,3})?", txt) or re.search(r"nasdaq|new york stock exchange|nyse", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
