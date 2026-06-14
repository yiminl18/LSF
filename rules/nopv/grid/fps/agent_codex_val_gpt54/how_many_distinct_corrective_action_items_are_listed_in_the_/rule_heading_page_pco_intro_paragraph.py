def rule_heading_page_pco_intro_paragraph(doc: dict) -> list[dict]:
    """Match the first text paragraph after the final Proposed Compliance Order heading page."""
    try:
        texts = doc.get("texts", [])
        exact_items = [
            s for s in texts
            if s.get("label") == "list_item"
            and (((s.get("structure") or {}).get("path_text") or "").strip().upper() == "PROPOSED COMPLIANCE ORDER")
        ]
        if exact_items:
            return []
        heading_index = None
        for i, span in enumerate(texts):
            if (span.get("text") or "").strip().upper() == "PROPOSED COMPLIANCE ORDER":
                heading_index = i
        if heading_index is None:
            return []
        heading_page = texts[heading_index].get("page_no")
        saw_list_item = False
        intro_span = None
        for span in texts[heading_index + 1:]:
            text = (span.get("text") or "").strip().lower()
            if span.get("page_no") != heading_page or text.startswith("response to this notice"):
                break
            if intro_span is None and span.get("label") == "text":
                intro_span = span
            if span.get("label") == "list_item":
                saw_list_item = True
        return [intro_span] if intro_span is not None and saw_list_item else []
    except Exception:
        return []
