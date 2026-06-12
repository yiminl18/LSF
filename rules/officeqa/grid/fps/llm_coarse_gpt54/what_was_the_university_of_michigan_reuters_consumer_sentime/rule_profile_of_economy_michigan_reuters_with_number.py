def rule_profile_of_economy_michigan_reuters_with_number(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans containing Michigan/Reuters naming and a likely numeric reading."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "profile of the economy" in path.lower():
                if re.search(r"(university of michigan|michigan/reuters|reuters)", text, re.I) and re.search(r"\b\d{2,3}\.?\d*\b", text):
                    out.append(span)
        return out
    except Exception:
        return []
