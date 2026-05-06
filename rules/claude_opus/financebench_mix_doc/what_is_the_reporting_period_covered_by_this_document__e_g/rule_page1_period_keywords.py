def rule_page1_period_keywords(doc: dict) -> list[dict]:
    """Match page 1 spans containing reporting period keywords (fiscal year, quarterly period, date of report)."""
    try:
        keywords = [
            "fiscal year ended",
            "quarterly period ended",
            "date of report",
            "date of earliest event",
        ]
        results = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            text = s.get("text", "").lower()
            text_span = s.get("text_span", "").lower()
            combined = text + " " + text_span
            if any(kw in combined for kw in keywords):
                results.append(s)
        return results
    except Exception:
        return []
