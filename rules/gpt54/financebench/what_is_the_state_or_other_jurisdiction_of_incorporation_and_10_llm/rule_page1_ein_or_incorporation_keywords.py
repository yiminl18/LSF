def rule_page1_ein_or_incorporation_keywords(doc: dict) -> list[dict]:
    """Match page-1 spans containing incorporation or IRS employer identification keywords."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(
                r"(state or other jurisdiction of incorporation|state of incorporation|irs employer identification|i\.r\.s\. employer identification|employer identification no)",
                text,
                re.I,
            ):
                out.append(span)
        return out
    except Exception:
        return []
