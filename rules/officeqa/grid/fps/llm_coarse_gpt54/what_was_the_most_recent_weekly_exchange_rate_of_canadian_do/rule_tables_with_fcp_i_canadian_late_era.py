def rule_tables_with_fcp_i_canadian_late_era(doc: dict) -> list[dict]:
    """Match later-era Canadian dollar tables where Canadian positions are SECTION I / FCP-I-*."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "section i" in txt and "canadian dollar positions" in txt:
                out.append(span)
            elif "fcp-i-1" in txt and "weekly report of major market participants" in txt:
                out.append(span)
        return out
    except Exception:
        return []
