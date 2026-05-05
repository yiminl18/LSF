def rule_page1_cover_answer_cluster(doc: dict) -> list[dict]:
    """Match the dense cluster of page-1 spans around state, EIN, address, and exact-name labels."""
    try:
        import re
        texts = doc.get("texts", [])
        hits = []
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(
                r"exact name of registrant|state of incorporation|state or other jurisdiction|employer identification|i\.r\.s\. employer identification|\d{2}-\d{7}",
                txt,
                re.I,
            ):
                hits.append(i)
        if not hits:
            return []
        start = max(0, min(hits) - 2)
        end = min(len(texts), max(hits) + 3)
        return [s for s in texts[start:end] if s.get("page_no") == 1]
    except Exception:
        return []
