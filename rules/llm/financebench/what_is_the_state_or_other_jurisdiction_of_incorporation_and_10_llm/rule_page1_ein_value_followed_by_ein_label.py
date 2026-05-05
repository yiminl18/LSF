def rule_page1_ein_value_followed_by_ein_label(doc: dict) -> list[dict]:
    """Match page-1 spans immediately preceding an IRS Employer Identification label span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            nxt_text = (nxt.get("text") or "").strip()
            cur_text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and nxt.get("page_no") == 1:
                if re.search(r"i\.r\.s\. employer identification no\.?|irs employer identification no\.?|employer identification no", nxt_text, re.I):
                    if re.fullmatch(r"\d{2}-\d{7}", cur_text):
                        out.append(span)
        return out
    except Exception:
        return []
