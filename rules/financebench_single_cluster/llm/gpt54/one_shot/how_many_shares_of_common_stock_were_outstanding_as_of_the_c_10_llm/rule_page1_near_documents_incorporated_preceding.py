def rule_page1_near_documents_incorporated_preceding(doc: dict) -> list[dict]:
    """Match spans immediately preceding 'DOCUMENTS INCORPORATED BY REFERENCE' that mention outstanding shares."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            t = (span.get("text") or "").lower()
            if "documents incorporated by reference" in t:
                for j in range(max(0, i - 3), i):
                    prev = texts[j]
                    pt = (prev.get("text") or "").lower()
                    if prev.get("page_no") in (1, 2) and ("outstanding" in pt or "common stock" in pt):
                        out.append(prev)
        return out
    except Exception:
        return []
