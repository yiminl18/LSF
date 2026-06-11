def rule_page1_spans_before_telephone_label(doc: dict) -> list[dict]:
    """Match page-1 spans in the company block before the telephone label, where state and EIN often appear."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and ("telephone number" in txt.lower() or "Registrant’s telephone number" in txt or "Registrant's telephone number" in txt):
                for prev in texts[max(0, i-12):i]:
                    if prev.get("page_no") == 1:
                        out.append(prev)
                break
        return out
    except Exception:
        return []
