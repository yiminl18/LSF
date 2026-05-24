def rule_page1_address_before_phone_number_same_span(doc: dict) -> list[dict]:
    """Match page 1 spans that contain both address and phone number, common in cover pages."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = span.get("text") or ""
            if re.search(r"\(\d{3}\)\s*\d{3}[-‑]\d{4}", txt) or re.search(r"\+\d{2}\s*\d+", txt):
                if re.search(r"\b[A-Z][a-zA-Z\.\- ]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)*)\b", txt):
                    out.append(span)
        return out
    except Exception:
        return []
