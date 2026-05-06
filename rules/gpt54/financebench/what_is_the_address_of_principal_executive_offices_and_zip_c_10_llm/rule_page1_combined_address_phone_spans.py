def rule_page1_combined_address_phone_spans(doc: dict) -> list[dict]:
    """Match page-1 spans combining address and phone in one line."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\(\d{3}\)\s*\d{3}[- ]?\d{4}", txt) and re.search(r"\b\d{5}(?:-\d{4})?\b", txt):
                out.append(span)
            elif "principal executive offices" in low and re.search(r"\(\d{3}\)\s*\d{3}[- ]?\d{4}", txt):
                out.append(span)
        return out
    except Exception:
        return []
