def rule_page1_window_after_ein_label(doc: dict) -> list[dict]:
    """Match a small page-1 window around the IRS Employer Identification label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r"i\.r\.s\. employer identification no\.?|irs employer identification no\.?|employer identification no", txt, re.I):
                for s in texts[max(0, i - 3): min(len(texts), i + 4)]:
                    if s.get("page_no") == 1:
                        out.append(s)
        return out
    except Exception:
        return []
