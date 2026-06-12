def rule_profile_of_economy_summary_indicators(doc: dict) -> list[dict]:
    """Match the Profile of the Economy summary indicators section where GDP growth is summarized."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Profile of the Economy" in path and re.search(r"Summary of Economic Indicators|summary of economic indicators", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
