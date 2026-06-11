def rule_page1_symbol_like_in_text(doc: dict) -> list[dict]:
    """Match page-1 spans containing ticker-like tokens near trading/listing language."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") != 1:
                continue
            if re.search(r"\b[A-Z]{1,5}(?:[/-][A-Z0-9]{1,5})?\d{0,2}\b", txt) and (
                re.search(r"trading symbol", txt, re.I)
                or re.search(r"exchange", txt, re.I)
                or re.search(r"section 12\(b\)", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
