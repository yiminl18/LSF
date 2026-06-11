def rule_form_any_label_exact(doc: dict) -> list[dict]:
    """Match any page-1 span whose text is exactly a common SEC form name."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.fullmatch(
                r"FORM\s+(10-K|10-Q|8-K|20-F|6-K|S-1|S-3|S-4|DEF 14A|SC 13D|SC 13G)", text, re.I
            ):
                out.append(span)
        return out
    except Exception:
        return []
