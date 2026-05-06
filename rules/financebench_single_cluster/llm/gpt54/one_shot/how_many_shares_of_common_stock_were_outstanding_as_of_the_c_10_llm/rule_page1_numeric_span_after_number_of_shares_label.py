def rule_page1_numeric_span_after_number_of_shares_label(doc: dict) -> list[dict]:
    """Match the numeric span immediately following a 'number of shares of common stock outstanding' label on page 1."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        num_re = re.compile(r"^\d[\d,]{5,}$")
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "number of shares of common stock outstanding" in text:
                for nxt in texts[i + 1:i + 4]:
                    nt = (nxt.get("text") or "").strip()
                    if nxt.get("page_no") == 1 and num_re.match(nt):
                        out.append(nxt)
    except Exception:
        return []
    return out
