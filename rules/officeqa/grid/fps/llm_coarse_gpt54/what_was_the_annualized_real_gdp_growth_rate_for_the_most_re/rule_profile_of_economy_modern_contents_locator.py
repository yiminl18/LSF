def rule_profile_of_economy_modern_contents_locator(doc: dict) -> list[dict]:
    """Match contents entries that point to the Profile of the Economy section in modern bulletins."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"PROFILE OF THE ECONOMY|Profile of the Economy", txt) and span.get("page_no") in {1, 4, 5, 9, 11}:
                out.append(span)
        return out
    except Exception:
        return []
