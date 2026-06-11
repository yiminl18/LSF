def rule_first_current_report_or_news_release(doc: dict) -> list[dict]:
    """Return the first page-1 span that says CURRENT REPORT or NEWS RELEASE."""
    try:
        candidates = []
        for i, span in enumerate(doc.get("texts", [])):
            text = (span.get("text") or "").strip().upper()
            if span.get("page_no") == 1 and (text == "CURRENT REPORT" or text == "NEWS RELEASE"):
                candidates.append((i, span))
        return [candidates[0][1]] if candidates else []
    except Exception:
        return []
