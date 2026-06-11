def rule_page1_h1_after_form_and_before_part_i(doc: dict) -> list[dict]:
    """Match page-1 H1 headers between the form title and later body sections like PART I."""
    try:
        texts = doc.get("texts", [])
        form_idx = None
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and "FORM 10-" in (s.get("text") or "").upper():
                form_idx = i
                break
        if form_idx is None:
            return []
        out = []
        for span in texts[form_idx + 1:]:
            if span.get("page_no") != 1:
                break
            if (
                span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "FORM 10-" not in (span.get("text") or "").upper()
            ):
                out.append(span)
        return out
    except Exception:
        return []
