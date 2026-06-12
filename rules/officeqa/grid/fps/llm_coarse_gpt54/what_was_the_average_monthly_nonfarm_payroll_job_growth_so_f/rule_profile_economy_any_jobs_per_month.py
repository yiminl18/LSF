def rule_profile_economy_any_jobs_per_month(doc: dict) -> list[dict]:
    """Match any Profile of the Economy text span mentioning jobs per month, for high recall."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "text"
            and "Profile of the Economy" in (((span.get("structure") or {}).get("path_text")) or "")
            and re.search(r'jobs per month', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
