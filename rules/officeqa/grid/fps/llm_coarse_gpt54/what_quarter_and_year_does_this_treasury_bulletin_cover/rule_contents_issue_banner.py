def rule_contents_issue_banner(doc: dict) -> list[dict]:
    """Match issue banner/date spans on contents pages, e.g. SUMMER ISSUE, SEPTEMBER 1989."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        pat = re.compile(
            r"\b("
            r"march|june|september|december|"
            r"spring issue|summer issue|fall issue|winter issue|"
            r"first quarter, fiscal \d{4}|second quarter, fiscal \d{4}|third quarter, fiscal \d{4}|fourth quarter, fiscal \d{4}"
            r")\b",
            re.I,
        )
        for s in texts:
            path = ((s.get("structure") or {}).get("path_text") or "").lower()
            txt = (s.get("text") or "").strip()
            if "contents" in path and txt and pat.search(txt):
                out.append(s)
        return out
    except Exception:
        return []
