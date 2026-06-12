def rule_canadian_weekly_report_tables(doc: dict) -> list[dict]:
    """Match later-era Canadian dollar position tables with weekly/monthly/quarterly participant reports."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "canadian dollar positions" in txt or ("fcp-i-1" in txt and "weekly report of major market participants" in txt):
                out.append(span)
        return out
    except Exception:
        return []
