def rule_modern_contents_month_banner(doc: dict) -> list[dict]:
    """Match modern contents-page month banners like 'DECEMBER 1997' or 'MARCH 2016'."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(r"^(MARCH|JUNE|SEPTEMBER|DECEMBER)\s+\d{4}$", re.I)
        out = []
        for s in texts:
            path = ((s.get("structure") or {}).get("path_text") or "").lower()
            txt = (s.get("text") or "").strip()
            if txt and pat.match(txt) and "contents" in path:
                out.append(s)
        return out
    except Exception:
        return []
