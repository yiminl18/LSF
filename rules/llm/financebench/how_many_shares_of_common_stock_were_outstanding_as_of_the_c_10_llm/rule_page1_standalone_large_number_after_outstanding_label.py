def rule_page1_standalone_large_number_after_outstanding_label(doc: dict) -> list[dict]:
    """Match standalone numeric spans on page 1 that follow a label like 'Number of shares of common stock outstanding as of ...'."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        num_re = re.compile(r"^\$?\s*\d[\d,]{5,}\s*$")
        for i, span in enumerate(texts):
            text = (span.get("text") or "").strip()
            if span.get("page_no") != 1 or not num_re.match(text):
                continue
            prev_window = texts[max(0, i - 4):i]
            prev_text = " ".join((p.get("text") or "") for p in prev_window).lower()
            if "number of shares of common stock outstanding" in prev_text or ("outstanding" in prev_text and "common stock" in prev_text):
                out.append(span)
    except Exception:
        return []
    return out
