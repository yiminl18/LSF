def rule_page1_phone_in_cover_metadata_window(doc: dict) -> list[dict]:
    """Match phone-like spans in the early cover-page metadata window before Item sections begin."""
    try:
        import re
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for idx, span in enumerate(doc.get("texts", [])):
            if idx > 120:
                break
            if span.get("page_no") == 1:
                txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
                if phone_pat.search(txt):
                    out.append(span)
        return out
    except Exception:
        return []
