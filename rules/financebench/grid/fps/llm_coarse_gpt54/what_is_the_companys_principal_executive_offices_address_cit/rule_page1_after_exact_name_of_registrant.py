def rule_page1_after_exact_name_of_registrant(doc: dict) -> list[dict]:
    """Match address-like spans appearing shortly after the exact-name-of-registrant label on page 1."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r'exact name of registrant', txt, re.I):
                for cand in texts[i+1:i+8]:
                    if cand.get("page_no") != 1:
                        continue
                    ctext = (cand.get("text") or "").strip()
                    if re.search(r'\d{1,6}\s+\S+', ctext) and (
                        re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', ctext) or
                        re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+', ctext) or
                        re.search(r'\bUnited Kingdom\b', ctext)
                    ):
                        out.append(cand)
        return out
    except Exception:
        return []
