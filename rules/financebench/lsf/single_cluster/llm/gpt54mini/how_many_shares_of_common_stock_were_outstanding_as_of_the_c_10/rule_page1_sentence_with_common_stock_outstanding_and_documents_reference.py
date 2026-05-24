def rule_page1_sentence_with_common_stock_outstanding_and_documents_reference(doc: dict) -> list[dict]:
    """Match page-1 spans with common-stock-outstanding language in the same large cover block as documents-incorporated text."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            full = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "common stock" in full and "outstanding" in full and "documents incorporated" in full:
                out.append(span)
    except Exception:
        return []
    return out
