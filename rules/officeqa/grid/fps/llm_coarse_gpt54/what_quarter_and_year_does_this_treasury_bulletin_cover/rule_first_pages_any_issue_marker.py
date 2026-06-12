def rule_first_pages_any_issue_marker(doc: dict) -> list[dict]:
    """Match any early-page span containing issue markers like Issue, Quarter, Fiscal, or month names."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b(issue|quarter|fiscal|january|february|march|april|may|june|july|august|september|october|november|december|spring|summer|fall|winter)\b",
            re.I,
        )
        return [
            s for s in texts
            if s.get("page_no", 999) <= 12 and pat.search((s.get("text") or "").strip())
        ]
    except Exception:
        return []
