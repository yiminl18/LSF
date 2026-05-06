def rule_page1_address_header_with_zip_in_textspan(doc: dict) -> list[dict]:
    """Match page 1 address headers whose text_span mentions zip code or address label."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            txt = span.get("text") or ""
            tsp = (span.get("text_span") or "").lower()
            if ("address of principal executive offices" in tsp or "zip code" in tsp) and any(ch.isdigit() for ch in txt):
                out.append(span)
        return out
    except Exception:
        return []
