def rule_profile_of_economy_modern_answer_pages(doc: dict) -> list[dict]:
    """Match likely answer spans on pages 7-11 in 21st-century bulletins."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            p = span.get("page_no")
            txt = (span.get("text") or "")
            if p is not None and 7 <= p <= 11 and re.search(r"real\s+GDP|gross\s+domestic\s+product|Economic Growth|Growth of Real GDP", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
