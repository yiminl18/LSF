def rule_profile_of_economy_unemployment_rate_on_profile_pages(doc: dict) -> list[dict]:
    """Match unemployment-related spans on the same pages as Profile of the Economy content."""
    import re
    out = []
    try:
        profile_pages = set()
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if "Profile of the Economy" in path or re.search(r"Profile of the Economy", text, re.I):
                if isinstance(span.get("page_no"), int):
                    profile_pages.add(span["page_no"])
        for span in doc.get("texts", []):
            if span.get("page_no") in profile_pages and span.get("label") == "text":
                text = span.get("text", "") or ""
                if re.search(r"unemployment rate", text, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
