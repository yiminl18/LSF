def rule_page1_cover_block_before_securities_registered(doc: dict) -> list[dict]:
    """Match spans in the cover block before 'Securities registered pursuant to Section 12(b)' where phone appears."""
    try:
        import re
        texts = doc.get("texts", [])
        cutoff = len(texts)
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and re.search(r"securities registered pursuant to section 12\(b\)", span.get("text", "") or "", re.I):
                cutoff = i
                break
        out = []
        for span in texts[:cutoff]:
            if span.get("page_no") != 1:
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"telephone number|area code", blob, re.I) or re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
