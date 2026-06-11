def rule_page1_address_label_spans(doc: dict) -> list[dict]:
    """Match page-1 spans containing the principal executive office address label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if re.search(r'address of principal executive offices', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
