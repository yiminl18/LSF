def rule_profile_economy_modern_answer_exact_pattern(doc: dict) -> list[dict]:
    """Match modern answer pattern 'In YYYY thus far, job growth has average[d] NNN,NNN'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if (
                span.get("label") == "text"
                and re.search(r'In\s+20\d{2}\s+thus far', txt, re.I)
                and re.search(r'job growth has average[d]?\s+\d[\d,]*', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
