def rule_page1_total_shares_after_class_a_class_b_without_outstanding(doc: dict) -> list[dict]:
    """Match the final total-share numeric span after Class A and Class B counts in dual-class cover layouts even if 'outstanding' is only in a nearby label."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        num_re = re.compile(r"^\d[\d,]{5,}$")
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if not num_re.match(txt):
                continue
            prev = " ".join((x.get("text") or "") for x in texts[max(0, i - 12):i]).lower()
            if "class a" in prev and "class b" in prev and "number of shares" in prev:
                out.append(span)
    except Exception:
        return []
    return out
