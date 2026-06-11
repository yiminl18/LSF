def rule_form_header_text_span_period(doc: dict) -> list[dict]:
    """Match section headers whose title is FORM 10-K/10-Q/8-K and whose text_span contains the reporting period."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                head = span.get("text") or ""
                tail = span.get("text_span") or ""
                if re.search(r'FORM\s+10-(K|Q|8-K)', head, re.I) and re.search(
                    r'(fiscal year ended|quarterly period ended|date of report|date of earliest event reported)',
                    tail,
                    re.I,
                ):
                    out.append(span)
        return out
    except Exception:
        return []
