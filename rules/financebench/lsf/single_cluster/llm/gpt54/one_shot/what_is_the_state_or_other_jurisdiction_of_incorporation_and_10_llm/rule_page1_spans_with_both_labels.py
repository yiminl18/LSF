def rule_page1_spans_with_both_labels(doc: dict) -> list[dict]:
    """Match page-1 spans containing both incorporation and IRS identification labels together."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(
                r"(state or other jurisdiction of incorporation|state of incorporation)",
                text,
                re.I,
            ) and re.search(
                r"(irs employer identification|i\.r\.s\. employer identification|employer identification no)",
                text,
                re.I,
            ):
                out.append(span)
        return out
    except Exception:
        return []
