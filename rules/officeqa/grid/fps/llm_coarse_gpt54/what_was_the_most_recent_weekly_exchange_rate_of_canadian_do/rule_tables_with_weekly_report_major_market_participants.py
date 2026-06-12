def rule_tables_with_weekly_report_major_market_participants(doc: dict) -> list[dict]:
    """Match later-era foreign currency tables using the 'Weekly Report of Major Market Participants' wording."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "weekly report of major market participants" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
