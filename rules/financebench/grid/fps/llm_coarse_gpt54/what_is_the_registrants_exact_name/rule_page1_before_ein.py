def rule_page1_before_ein(doc: dict) -> list[dict]:
    """Match the prominent span immediately before an IRS Employer Identification label on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            low = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "i.r.s. employer identification" in low or "irs employer identification" in low:
                for j in range(max(0, i - 5), i):
                    prev = texts[j]
                    if prev.get("page_no") == 1 and prev.get("bold") == 1 and float(prev.get("size") or 0) >= 10:
                        ptxt = (prev.get("text") or "").lower()
                        if "exact name of registrant" not in ptxt and "commission file" not in ptxt:
                            out.append(prev)
                break
        return out
    except Exception:
        return []
