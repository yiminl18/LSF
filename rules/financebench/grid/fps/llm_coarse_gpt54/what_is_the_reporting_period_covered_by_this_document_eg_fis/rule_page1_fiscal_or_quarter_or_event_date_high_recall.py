def rule_page1_fiscal_or_quarter_or_event_date_high_recall(doc: dict) -> list[dict]:
    """High-recall page-1 rule for any span mentioning fiscal year, quarter end, or event date."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").lower()
            if any(k in text for k in [
                "fiscal year ended",
                "quarterly period ended",
                "quarter ended",
                "date of report",
                "date of earliest event reported",
                "event date",
                "event reported",
                "period ending"
            ]):
                out.append(span)
        return out
    except Exception:
        return []
