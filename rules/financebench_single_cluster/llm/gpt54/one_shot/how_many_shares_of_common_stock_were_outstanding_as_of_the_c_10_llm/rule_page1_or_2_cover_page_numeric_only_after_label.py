def rule_page1_or_2_cover_page_numeric_only_after_label(doc: dict) -> list[dict]:
    """Match standalone numeric spans on page 1 or 2 that are near outstanding-share labels."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        num_re = re.compile(r"^\d[\d,]{5,}$")
        for i, span in enumerate(texts):
            if span.get("page_no") not in {1, 2}:
                continue
            text = (span.get("text") or "").strip()
            if not num_re.match(text):
                continue
            ctx = " ".join((x.get("text") or "") for x in texts[max(0, i - 6):min(len(texts), i + 1)]).lower()
            if "outstanding" in ctx and ("common stock" in ctx or "shares" in ctx):
                out.append(span)
    except Exception:
        return []
    return out
