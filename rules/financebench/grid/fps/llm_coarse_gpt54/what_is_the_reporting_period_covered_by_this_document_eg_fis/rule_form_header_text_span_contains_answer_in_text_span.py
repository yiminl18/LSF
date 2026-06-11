def rule_form_header_text_span_contains_answer_in_text_span(doc: dict) -> list[dict]:
    """Match spans where the answer phrase appears in text_span metadata of a top form header."""
    try:
        out = []
        for span in doc.get("texts", []):
            tspan = (span.get("text_span") or "").lower()
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "form " in text:
                if any(k in tspan for k in [
                    "fiscal year ended",
                    "quarterly period ended",
                    "date of report",
                    "date of earliest event reported"
                ]):
                    out.append(span)
        return out
    except Exception:
        return []
