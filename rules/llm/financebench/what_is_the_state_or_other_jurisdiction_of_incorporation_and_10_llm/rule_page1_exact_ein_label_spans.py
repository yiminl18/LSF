def rule_page1_exact_ein_label_spans(doc: dict) -> list[dict]:
    """Match page-1 spans whose text is the IRS Employer Identification label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(
                r"^\(?\s*(i\.r\.s\. employer identification no\.?|irs employer identification no\.?)\s*\)?$",
                text,
                re.I,
            ):
                out.append(span)
        return out
    except Exception:
        return []
