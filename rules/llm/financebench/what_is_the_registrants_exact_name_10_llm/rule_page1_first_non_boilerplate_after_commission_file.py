def rule_page1_first_non_boilerplate_after_commission_file(doc: dict) -> list[dict]:
    """Match the first bold alphabetic span after the commission file number on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen = False
        for span in texts:
            if span.get("page_no") != 1:
                continue
            low = (span.get("text") or "").lower()
            if "commission file" in low:
                seen = True
                continue
            if not seen:
                continue
            if span.get("bold") != 1:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if "exact name of registrant" in low:
                continue
            if "state or other jurisdiction" in low:
                continue
            if any(ch.isalpha() for ch in txt):
                out.append(span)
                break
        return out
    except Exception:
        return []
