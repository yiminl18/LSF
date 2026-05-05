def rule_page1_before_documents_incorporated(doc: dict) -> list[dict]:
    """Match page-1 spans shortly before the 'DOCUMENTS INCORPORATED BY REFERENCE' marker."""
    try:
        spans = doc.get("texts", [])
        out = []
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "documents incorporated by reference" in txt:
                for j in range(max(0, i - 8), i):
                    if spans[j].get("page_no") == 1:
                        out.append(spans[j])
        return out
    except Exception:
        return []
