def rule_page1_cover_block_with_documents_incorporated(doc: dict) -> list[dict]:
    """Match page-1/2 cover-page spans near 'DOCUMENTS INCORPORATED BY REFERENCE' that also mention outstanding shares."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            t = text.lower()
            if span.get("page_no") in (1, 2):
                if "documents incorporated by reference" in t and ("outstanding" in t or "common stock" in t):
                    out.append(span)
        return out
    except Exception:
        return []
