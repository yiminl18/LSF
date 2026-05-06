def rule_phone_near_exact_name(doc: dict) -> list[dict]:
    """Match spans near the exact-name-of-registrant label that contain or neighbor the phone number."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"exact name of registrant", text, re.I):
                for j in range(max(0, i - 1), min(len(texts), i + 15)):
                    cand = texts[j]
                    ctext = ((cand.get("text") or "") + " " + (cand.get("text_span") or "")).strip()
                    if cand.get("page_no") == 1 and re.search(r"telephone|area code|(\(\d{3}\)\s*\d{3}[-\s]?\d{4})|(\+\d{1,3}\s*\d)", ctext, re.I):
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
