def rule_page12_form_path_period(doc: dict) -> list[dict]:
    """Match page-1/2 FORM-anchored cover spans whose text contains period cues."""
    try:
        import re
        return [
            s for s in doc.get("texts", [])
            if (s.get("page_no") or 0) <= 2
            and s.get("label") in {"text", "section_header", "checkbox_selected"}
            and "form 10-" in (((s.get("structure") or {}).get("path_text")) or "").lower()
            and re.search(r"\bended\b|date of report", (s.get("text") or "").lower())
        ]
    except Exception:
        return []
