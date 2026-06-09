def rule_page12_bold_exact_form_8k(doc: dict) -> list[dict]:
    """Match bold page-1/2 cover spans whose text is exactly FORM 8-K."""
    try:
        return [
            s for s in doc.get("texts", [])
            if (s.get("page_no") or 99) <= 2
            and s.get("label") in {"text", "section_header"}
            and s.get("bold") == 1
            and (s.get("text") or "").strip().upper() == "FORM 8-K"
        ]
    except Exception:
        return []
