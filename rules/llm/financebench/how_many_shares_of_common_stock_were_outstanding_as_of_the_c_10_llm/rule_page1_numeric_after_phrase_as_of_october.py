def rule_page1_numeric_after_phrase_as_of_october(doc: dict) -> list[dict]:
    """Match numeric spans on page 1 after labels ending with 'as of October ...'."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        num_re = re.compile(r"^\d[\d,]{5,}$")
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1 or not num_re.match(txt):
                continue
            prev = " ".join((x.get("text") or "") for x in texts[max(0, i - 6):i]).lower()
            if "as of october" in prev and ("outstanding" in prev or "number of shares" in prev):
                out.append(span)
    except Exception:
        return []
    return out
