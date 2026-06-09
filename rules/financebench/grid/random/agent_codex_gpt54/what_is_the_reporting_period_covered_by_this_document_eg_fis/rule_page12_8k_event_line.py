def rule_page12_8k_event_line(doc: dict) -> list[dict]:
    """Match page-1/2 8-K cover lines naming the earliest event reported date."""
    try:
        return [
            s for s in doc.get("texts", [])
            if (s.get("page_no") or 0) <= 2
            and s.get("label") in {"text", "section_header"}
            and "date of report" in (s.get("text") or "").lower()
            and "earliest event reported" in (s.get("text") or "").lower()
        ]
    except Exception:
        return []
