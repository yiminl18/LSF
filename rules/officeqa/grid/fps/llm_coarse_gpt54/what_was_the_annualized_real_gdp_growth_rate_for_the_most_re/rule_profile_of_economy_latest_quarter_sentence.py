def rule_profile_of_economy_latest_quarter_sentence(doc: dict) -> list[dict]:
    """Match sentences describing the most recent quarter's GDP growth in the economy profile."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"(latest|first|second|third|fourth)\s+quarter", txt, re.I) and \
               re.search(r"(real\s+GDP|gross\s+domestic\s+product|GDP)", txt, re.I) and \
               re.search(r"\d+\.\d+\s*percent", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
