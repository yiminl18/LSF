def rule_page1_before_securities_registered_section(doc: dict) -> list[dict]:
    """Match page-1 spans before the 'Securities registered pursuant to Section 12(b)' line."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and "Securities registered pursuant to Section 12(b)" in txt:
                for prev in texts[max(0, i-15):i]:
                    if prev.get("page_no") == 1:
                        out.append(prev)
                break
        return out
    except Exception:
        return []
