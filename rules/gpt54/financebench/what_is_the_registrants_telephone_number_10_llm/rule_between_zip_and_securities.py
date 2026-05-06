def rule_between_zip_and_securities(doc: dict) -> list[dict]:
    """Match page-1 spans between zip-code and securities-registered labels, a common phone location."""
    try:
        import re
        texts = doc.get("texts", [])
        zip_idxs = []
        sec_idxs = []
        for i, span in enumerate(texts):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"\(zip code\)|\bzip code\b", text, re.I):
                zip_idxs.append(i)
            if span.get("page_no") == 1 and re.search(r"securities registered pursuant to section 12\(b\)", text, re.I):
                sec_idxs.append(i)
        out = []
        for zi in zip_idxs:
            for si in sec_idxs:
                if zi <= si and si - zi <= 6:
                    for j in range(zi, si + 1):
                        if texts[j].get("page_no") == 1:
                            out.append(texts[j])
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
