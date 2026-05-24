def rule_page1_before_ein_marker(doc: dict) -> list[dict]:
    """Match page-1 spans occurring shortly before an '(I.R.S. Employer Identification No.)' marker."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = span.get("text") or ""
            if span.get("page_no") == 1 and "(I.R.S. Employer Identification No.)" in txt:
                for k in range(max(0, i - 4), i):
                    prev = texts[k]
                    if prev.get("page_no") == 1:
                        out.append(prev)
        return out
    except Exception:
        return []
