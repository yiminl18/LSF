def rule_subject_heading_with_long_summary_textspan(doc: dict) -> list[dict]:
    """Match heading spans whose text_span contains a long summary paragraph, indicating a summary subject heading."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            text_span = (span.get("text_span") or "").strip()
            up = txt.upper()
            if span.get("label") == "section_header" and txt and len(text_span) > 120:
                if "COUNSEL" in up or "OPINION" in up or "BACKGROUND" in up or "SUMMARY" in up:
                    continue
                if span.get("page_no") in {1, 2, 3}:
                    out.append(span)
        return out
    except Exception:
        return []
