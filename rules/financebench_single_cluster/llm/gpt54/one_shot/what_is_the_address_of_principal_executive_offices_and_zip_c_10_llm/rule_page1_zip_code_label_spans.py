def rule_page1_zip_code_label_spans(doc: dict) -> list[dict]:
    """Match page-1 spans containing ZIP code labels near the address."""
    try:
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            low = txt.lower()
            if "(zip code)" in low or "zip code" in low:
                out.append(span)
        return out
    except Exception:
        return []
