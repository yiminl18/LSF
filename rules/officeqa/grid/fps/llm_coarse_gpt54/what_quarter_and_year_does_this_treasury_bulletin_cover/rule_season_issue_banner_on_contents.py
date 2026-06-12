def rule_season_issue_banner_on_contents(doc: dict) -> list[dict]:
    """Match seasonal issue banners on contents pages in quarterly-era bulletins."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b(Spring|Summer|Fall|Winter)\s+Issue\b.*(?:\b\d{4}\b|\b(FIRST|SECOND|THIRD|FOURTH)\s+QUARTER\b)?",
            re.I,
        )
        out = []
        for s in texts:
            path = ((s.get("structure") or {}).get("path_text") or "").lower()
            txt = (s.get("text") or "").strip()
            if txt and pat.search(txt) and "contents" in path:
                out.append(s)
        return out
    except Exception:
        return []
