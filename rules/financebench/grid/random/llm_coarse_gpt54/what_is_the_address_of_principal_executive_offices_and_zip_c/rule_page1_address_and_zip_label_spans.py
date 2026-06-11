def rule_page1_address_and_zip_label_spans(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning both address and zip code labels."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if re.search(r'address of principal executive offices', txt, re.I) and re.search(r'zip code', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
