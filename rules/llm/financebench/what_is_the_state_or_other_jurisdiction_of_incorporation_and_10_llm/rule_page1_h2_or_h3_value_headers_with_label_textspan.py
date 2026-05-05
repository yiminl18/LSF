def rule_page1_h2_or_h3_value_headers_with_label_textspan(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 headers where the value is in text and the label is in text_span."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            level = span.get("structure", {}).get("level")
            if span.get("page_no") == 1 and span.get("label") == "section_header" and level in ("H2", "H3"):
                t = (span.get("text") or "").strip()
                ts = (span.get("text_span") or "").strip()
                if ts and re.search(r"state|jurisdiction|employer identification|i\.r\.s\.", ts, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
