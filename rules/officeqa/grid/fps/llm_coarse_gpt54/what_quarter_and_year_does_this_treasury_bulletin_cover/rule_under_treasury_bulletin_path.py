def rule_under_treasury_bulletin_path(doc: dict) -> list[dict]:
    """Match date-like spans whose structure path_text contains Treasury Bulletin."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Spring|Summer|Fall|Winter"
            r")\b",
            re.I,
        )
        out = []
        for s in texts:
            path = ((s.get("structure") or {}).get("path_text") or "")
            txt = (s.get("text") or "").strip()
            if "treasury bulletin" in path.lower() and txt and pat.search(txt):
                out.append(s)
        return out
    except Exception:
        return []
