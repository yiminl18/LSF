def rule_section_header_value_with_label_in_text_span(doc: dict) -> list[dict]:
    """Match section_header spans whose text is the value and text_span contains the explanatory label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            tspan = (span.get("text_span") or "").strip()
            if re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", tspan, re.I):
                out.append(span)
            elif re.search(r"(i\.?r\.?s\.?\s+)?employer\s+identification", tspan, re.I):
                out.append(span)
            elif re.fullmatch(r"\d{2}-\d{7}", txt):
                out.append(span)
        return out
    except Exception:
        return []
