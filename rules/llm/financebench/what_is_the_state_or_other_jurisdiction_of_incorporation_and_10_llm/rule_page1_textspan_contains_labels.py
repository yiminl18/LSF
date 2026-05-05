def rule_page1_textspan_contains_labels(doc: dict) -> list[dict]:
    """Match page-1 spans where text_span contains the incorporation/EIN labels."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            ts = (span.get("text_span") or "").strip()
            if span.get("page_no") == 1 and re.search(
                r"(state or other jurisdiction of incorporation|state of incorporation|i\.r\.s\. employer identification|irs employer identification|employer identification no)",
                ts,
                re.I,
            ):
                out.append(span)
        return out
    except Exception:
        return []
