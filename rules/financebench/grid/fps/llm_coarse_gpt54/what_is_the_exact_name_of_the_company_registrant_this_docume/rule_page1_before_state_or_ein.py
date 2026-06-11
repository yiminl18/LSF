def rule_page1_before_state_or_ein(doc: dict) -> list[dict]:
    """Match page-1 bold company-like span immediately preceding state/EIN/address cover details."""
    try:
        texts = doc.get("texts", [])
        out = []
        keywords = [
            "state or other jurisdiction of incorporation",
            "i.r.s. employer identification no",
            "irs employer identification no",
            "commission file number",
            "address of principal executive offices"
        ]
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            if span.get("bold", 0) != 1:
                continue
            if float(span.get("size", 0) or 0) < 10:
                continue
            low = txt.lower()
            if "form 10-" in low or "form 8-k" in low or "commission" in low or "current report" in low:
                continue
            window = texts[i:i+8]
            if any(any(k in ((w.get("text", "") or "") + " " + (w.get("text_span", "") or "")).lower() for k in keywords) for w in window):
                out.append(span)
        return out
    except Exception:
        return []
