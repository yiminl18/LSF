def rule_phone_before_table_of_contents(doc: dict) -> list[dict]:
    """Match phone-related spans appearing before the table of contents begins."""
    try:
        import re
        texts = doc.get("texts", [])
        toc_idx = None
        for i, span in enumerate(texts):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"table of contents|index to the form 10-k|\bindex\b", text, re.I):
                toc_idx = i
                break
        if toc_idx is None:
            toc_idx = len(texts)
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}-\d{3}-\d{4}\b)")
        for span in texts[:toc_idx]:
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and (phone_re.search(text) or re.search(r"registrant[’'`s]{0,2}\s+telephone", text, re.I)):
                out.append(span)
        return out
    except Exception:
        return []
