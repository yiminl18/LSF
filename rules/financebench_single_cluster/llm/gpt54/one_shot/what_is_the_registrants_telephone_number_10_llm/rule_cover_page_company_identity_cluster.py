def rule_cover_page_company_identity_cluster(doc: dict) -> list[dict]:
    """Match the cover-page company identity cluster around exact name, address, EIN, zip, and phone."""
    try:
        import re
        texts = doc.get("texts", [])
        anchor_idxs = []
        for i, span in enumerate(texts):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"exact name of registrant", text, re.I):
                anchor_idxs.append(i)
        out = []
        for i in anchor_idxs:
            for j in range(i, min(len(texts), i + 12)):
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
