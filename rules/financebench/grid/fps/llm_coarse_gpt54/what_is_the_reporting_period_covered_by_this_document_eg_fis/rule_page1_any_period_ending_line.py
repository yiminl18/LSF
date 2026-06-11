def rule_page1_any_period_ending_line(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'period ending' or 'period ended'."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and any(k in (span.get("text") or "").lower() for k in ["period ending", "period ended"])
        ]
    except Exception:
        return []
