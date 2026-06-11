def rule_form10q_h1_with_embedded_quarter_end(doc: dict) -> list[dict]:
    """Match H1 FORM 10-Q spans whose text_span embeds the quarterly period ended phrase."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and ((span.get("structure") or {}).get("path_text") or "").upper().startswith("FORM 10-Q")
            and "quarterly period ended" in (span.get("text_span") or "").lower()
        ]
    except Exception:
        return []
