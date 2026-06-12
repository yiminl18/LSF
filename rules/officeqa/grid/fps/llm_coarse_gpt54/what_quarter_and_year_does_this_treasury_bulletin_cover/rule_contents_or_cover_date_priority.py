def rule_contents_or_cover_date_priority(doc: dict) -> list[dict]:
    """Match date-like spans on either cover/title pages or contents pages, the two dominant locations."""
    import re
    try:
        texts = doc.get("texts", [])
        candidate_pages = set()
        for s in texts:
            txt = (s.get("text") or "").strip().lower()
            if "treasury bulletin" in txt or txt in {"contents", "table of contents"}:
                candidate_pages.add(s.get("page_no"))
        pat = re.compile(
            r"\b("
            r"march|june|september|december|january|february|april|may|july|august|october|november|"
            r"spring|summer|fall|winter|fiscal"
            r")\b",
            re.I,
        )
        return [
            s for s in texts
            if s.get("page_no") in candidate_pages and pat.search((s.get("text") or "").strip())
        ]
    except Exception:
        return []
