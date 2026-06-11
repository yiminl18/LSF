def rule_page1_cover_block_after_ein(doc: dict) -> list[dict]:
    """Match address-like spans in the cover block near IRS Employer Identification references."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r'irs employer identification|employer identification no', txt, re.I):
                for cand in texts[max(0, i-4):i+5]:
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
