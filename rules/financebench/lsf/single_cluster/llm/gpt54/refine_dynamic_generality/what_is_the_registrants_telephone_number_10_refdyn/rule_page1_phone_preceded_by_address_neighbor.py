def rule_page1_phone_preceded_by_address_neighbor(doc: dict) -> list[dict]:
    """Match phone spans preceded by address/zip/EIN cover-page neighbors."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if not phone_re.search(text):
                continue
            prev = texts[max(0, i-4):i]
            prev_text = " ".join(((p.get("text") or "") + " " + (p.get("text_span") or "")) for p in prev).lower()
            if "address of principal executive offices" in prev_text or "zip code" in prev_text or "employer identification" in prev_text:
                out.append(span)
        return out
    except Exception:
        return []
