def rule_page13_8k_event_report_date_spans(doc: dict) -> list[dict]:
    """Match early-page 8-K cover spans that state the date of report and earliest event."""
    try:
        import re

        results = []
        month_date_re = (
            r"(january|february|march|april|may|june|july|august|september|"
            r"october|november|december)\s+\d{1,2},\s+\d{4}"
        )

        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 3:
                continue
            if span.get("label") not in ("text", "section_header"):
                continue
            lowered = " ".join(span.get("text", "").lower().split())
            if (
                lowered.startswith("date of report")
                and "event reported" in lowered
                and re.search(month_date_re, lowered)
            ):
                results.append(span)
        return results
    except Exception:
        return []
