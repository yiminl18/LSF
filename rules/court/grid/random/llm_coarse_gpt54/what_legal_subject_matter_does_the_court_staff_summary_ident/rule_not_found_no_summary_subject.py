def rule_not_found_no_summary_subject(doc: dict) -> list[dict]:
    """Return no spans when the document appears to lack a court-staff summary subject heading."""
    try:
        texts = doc.get("texts", [])
        has_summary = any("SUMMARY" in ((s.get("text") or "").upper()) for s in texts)
        if not has_summary:
            return []
        generic_only = True
        for s in texts:
            txt = (s.get("text") or "").strip().upper()
            if s.get("bold") == 1 and s.get("page_no") in {1, 2, 3}:
                if all(bad not in txt for bad in ["SUMMARY", "COUNSEL", "OPINION", "ORDER", "BACKGROUND", "FILED"]):
                    generic_only = False
                    break
        return [] if generic_only else []
    except Exception:
        return []
