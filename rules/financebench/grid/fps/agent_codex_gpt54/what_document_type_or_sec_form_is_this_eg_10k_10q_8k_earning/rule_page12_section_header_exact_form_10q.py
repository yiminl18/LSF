def rule_page12_section_header_exact_form_10q(doc: dict) -> list[dict]:
    """Match page-1/2 section headers whose cover text is exactly FORM 10-Q."""
    try:
        return [
            s for s in doc.get("texts", [])
            if (s.get("page_no") or 99) <= 2
            and s.get("label") == "section_header"
            and (s.get("text") or "").strip().upper() == "FORM 10-Q"
        ]
    except Exception:
        return []
