def rule_form_heading_with_exact_common_answers(doc: dict) -> list[dict]:
    """Match spans whose text exactly equals one of the common answer strings in this corpus."""
    try:
        targets = {
            "form 10-k",
            "form 10-q",
            "form 8-k",
            "current report",
            "news release",
        }
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip().lower()
            if txt in targets:
                out.append(span)
        return out
    except Exception:
        return []
