def rule_section_header_text_span_reporting_period(doc: dict) -> list[dict]:
    """Match section_header spans whose text itself contains the reporting period phrase."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "section_header" and any(
                k in (span.get("text") or "").lower()
                for k in ["fiscal year ended", "quarterly period ended", "date of report", "date of earliest event reported"]
            )
        ]
    except Exception:
        return []
