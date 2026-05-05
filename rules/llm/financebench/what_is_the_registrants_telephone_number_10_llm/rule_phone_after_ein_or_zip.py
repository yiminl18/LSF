def rule_phone_after_ein_or_zip(doc: dict) -> list[dict]:
    """Match spans shortly after EIN or zip-code labels, where the phone number often follows."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"employer identification|zip code", text, re.I):
                for j in range(i, min(len(texts), i + 5)):
                    cand = texts[j]
                    ctext = ((cand.get("text") or "") + " " + (cand.get("text_span") or "")).strip()
                    if cand.get("page_no") == 1 and re.search(r"telephone|area code|(\(\d{3}\)\s*\d{3}[-\s]?\d{4})|(\+\d{1,3}\s*\d)|\b\d{3}-\d{3}-\d{4}\b", ctext, re.I):
                        out.append(cand)
        seen = set()
        dedup = []
        for s in out:
            key = id(s)
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        return dedup
    except Exception:
        return []
