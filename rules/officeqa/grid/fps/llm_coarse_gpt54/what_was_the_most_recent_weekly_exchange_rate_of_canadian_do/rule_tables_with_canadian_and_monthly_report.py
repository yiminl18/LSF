def rule_tables_with_canadian_and_monthly_report(doc: dict) -> list[dict]:
    """Match Canadian dollar position tables mentioning monthly report of major market participants."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "canadian" in txt and "monthly report of major market participants" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
