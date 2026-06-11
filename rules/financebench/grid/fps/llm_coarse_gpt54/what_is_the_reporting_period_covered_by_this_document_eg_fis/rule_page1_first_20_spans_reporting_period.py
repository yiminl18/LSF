def rule_page1_first_20_spans_reporting_period(doc: dict) -> list[dict]:
    """Match reporting-period spans among the first 20 spans, reflecting cover-page placement."""
    try:
        out = []
        for span in doc.get("texts", [])[:20]:
            text = (span.get("text") or "").lower()
            if any(k in text for k in [
                "fiscal year ended",
                "quarterly period ended",
                "date of report",
                "date of earliest event reported",
                "for the period ending"
            ]):
                out.append(span)
        return out
    except Exception:
        return []
