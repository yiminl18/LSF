def rule_form8k_h1_with_embedded_date_of_report(doc: dict) -> list[dict]:
    """Match H1 FORM 8-K spans whose text_span embeds the date-of-report phrase."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and ((span.get("structure") or {}).get("path_text") or "").upper().startswith("FORM 8-K")
            and "date of report" in (span.get("text_span") or "").lower()
        ]
    except Exception:
        return []
