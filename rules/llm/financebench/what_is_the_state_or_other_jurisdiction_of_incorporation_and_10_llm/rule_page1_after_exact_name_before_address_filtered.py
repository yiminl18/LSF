def rule_page1_after_exact_name_before_address_filtered(doc: dict) -> list[dict]:
    """Match likely answer spans between exact-name and address labels, filtering to state/EIN-like values and labels."""
    try:
        import re
        texts = doc.get("texts", [])
        start = None
        end = None
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if start is None and span.get("page_no") == 1 and "exact name of registrant" in txt:
                start = i
            if start is not None and end is None and span.get("page_no") == 1 and "address of principal executive offices" in txt:
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 20)
        out = []
        for s in texts[start:end]:
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).strip()
            if re.search(r"\d{2}-\d{7}|state|jurisdiction|incorporation|employer identification", txt, re.I):
                out.append(s)
        return out
    except Exception:
        return []
