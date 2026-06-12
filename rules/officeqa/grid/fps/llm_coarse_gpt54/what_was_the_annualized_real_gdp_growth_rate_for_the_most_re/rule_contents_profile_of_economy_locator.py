def rule_contents_profile_of_economy_locator(doc: dict) -> list[dict]:
    """Match contents spans that identify where the Profile of the Economy section starts."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Contents" in path and re.search(r"PROFILE OF THE ECONOMY|Profile of the Economy", txt):
                out.append(span)
        return out
    except Exception:
        return []
