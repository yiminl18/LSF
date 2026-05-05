def rule_page1_outstanding_near_documents_incorporated(doc: dict) -> list[dict]:
    """Match page-1 spans with outstanding-share language that often appear just before 'DOCUMENTS INCORPORATED BY REFERENCE'."""
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "outstanding" in text and ("common stock" in text or "shares" in text):
                window = texts[i + 1:i + 5]
                if any("documents incorporated" in (w.get("text") or "").lower() for w in window):
                    out.append(span)
    except Exception:
        return []
    return out
