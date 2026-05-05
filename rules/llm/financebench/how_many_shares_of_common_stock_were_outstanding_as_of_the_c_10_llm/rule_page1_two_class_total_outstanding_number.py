def rule_page1_two_class_total_outstanding_number(doc: dict) -> list[dict]:
    """Match page-1 standalone total-share number in dual-class layouts after Class A/Class B counts."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        num_re = re.compile(r"^\d[\d,]{5,}$")
        for i, span in enumerate(texts):
            text = (span.get("text") or "").strip()
            if span.get("page_no") != 1 or not num_re.match(text):
                continue
            prev_window = texts[max(0, i - 8):i]
            prev_text = " ".join((p.get("text") or "") for p in prev_window).lower()
            if "class a" in prev_text and "class b" in prev_text and "number of shares" in prev_text:
                out.append(span)
    except Exception:
        return []
    return out
