def rule_tables_with_canadian_and_quarterly_report(doc: dict) -> list[dict]:
    """Match Canadian dollar position tables mentioning quarterly report of large market participants."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "canadian" in txt and "quarterly report of large market participants" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
