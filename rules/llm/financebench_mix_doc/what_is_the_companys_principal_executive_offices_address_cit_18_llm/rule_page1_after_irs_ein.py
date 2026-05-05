def rule_page1_after_irs_ein(doc: dict) -> list[dict]:
    """Match spans on page 1 immediately after the I.R.S. Employer Identification No. span."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "i.r.s. employer identification no." in txt or "irs employer identification no." in txt:
                for j in range(i + 1, min(len(texts), i + 4)):
                    cand = texts[j]
                    if cand.get("page_no") == 1:
                        out.append(cand)
        return out
    except Exception:
        return []
