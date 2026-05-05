def rule_page1_h1_or_h2_with_phone_and_securities(doc: dict) -> list[dict]:
    """Match page-1 H1/H2 cover headers containing both phone and securities-registration text."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            level = ((span.get("structure") or {}).get("level") or "")
            if span.get("page_no") == 1 and level in {"H1", "H2"}:
                text = (span.get("text") or "") + " " + (span.get("text_span") or "")
                low = text.lower()
                if phone_re.search(text) and "securities registered pursuant to section 12" in low:
                    out.append(span)
        return out
    except Exception:
        return []
