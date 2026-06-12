def rule_profile_of_economy_recent_vintage_pages(doc: dict) -> list[dict]:
    """Match likely answer spans on pages 5-11 in modern bulletins where Profile of the Economy appears."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = (span.get("text") or "")
            if page is not None and 5 <= page <= 11:
                if re.search(r"real\s+GDP|gross\s+domestic\s+product|Growth of Real GDP|Economic Growth", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
