def rule_main_header_block_first_40_spans(doc: dict) -> list[dict]:
    """Match address-like spans in the first 40 spans, where SEC cover-page metadata usually appears."""
    try:
        import re
        spans = doc.get("texts", [])[:40]
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            low = txt.lower()
            if re.search(r"\b\d{5}(?:-\d{4})?\b", txt) and (
                re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low)
                or "address of principal executive offices" in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
