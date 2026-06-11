def rule_form10k_h1_with_embedded_fiscal_year(doc: dict) -> list[dict]:
    """Match H1 FORM 10-K spans whose text_span embeds the fiscal year ended phrase."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and ((span.get("structure") or {}).get("path_text") or "").upper().startswith("FORM 10-K")
            and "fiscal year ended" in (span.get("text_span") or "").lower()
        ]
    except Exception:
        return []
