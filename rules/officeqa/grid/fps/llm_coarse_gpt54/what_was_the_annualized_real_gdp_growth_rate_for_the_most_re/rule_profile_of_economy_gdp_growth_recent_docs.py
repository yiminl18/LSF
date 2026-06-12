def rule_profile_of_economy_gdp_growth_recent_docs(doc: dict) -> list[dict]:
    """Broad high-recall rule for modern docs: GDP-growth prose in Profile of the Economy."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Profile of the Economy" in path and re.search(r"(real\s+GDP|gross\s+domestic\s+product|Growth of Real GDP|Economic Growth)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
