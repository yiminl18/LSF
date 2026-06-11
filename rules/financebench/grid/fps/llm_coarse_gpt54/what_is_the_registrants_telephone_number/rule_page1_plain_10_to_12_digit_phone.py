def rule_page1_plain_10_to_12_digit_phone(doc: dict) -> list[dict]:
    """Match page-1 spans containing plain 10-12 digit phone strings without punctuation."""
    try:
        import re
        out = []
        pat = re.compile(r"\b\d{10,12}\b")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
                if pat.search(txt):
                    out.append(span)
        return out
    except Exception:
        return []
