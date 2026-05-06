def rule_page1_address_candidates_excluding_ir(doc: dict) -> list[dict]:
    """Match page-1 address candidates while excluding later investor-relations or mailing addresses."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            low = txt.lower()
            if any(k in low for k in ["attention:", "investor relations", "corporate secretary", "public reference room"]):
                continue
            if re.search(r"\b\d{1,6}\b", txt) and (
                re.search(r"\b(avenue|drive|road|plaza|street|way)\b", low)
                or re.search(r"\b\d{5}(?:-\d{4})?\b", txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
