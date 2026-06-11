def rule_page1_cover_block_before_documents_incorporated(doc: dict) -> list[dict]:
    """Match page-1 cover-block spans before 'DOCUMENTS INCORPORATED BY REFERENCE'."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'DOCUMENTS INCORPORATED BY REFERENCE', txt, re.I):
                break
            out.append(span)
        return out
    except Exception:
        return []
