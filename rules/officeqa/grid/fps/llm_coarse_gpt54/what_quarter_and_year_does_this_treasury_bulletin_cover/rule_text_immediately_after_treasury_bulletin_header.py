def rule_text_immediately_after_treasury_bulletin_header(doc: dict) -> list[dict]:
    """Match the text span immediately following a Treasury Bulletin title/header when it looks like the issue date."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        date_pat = re.compile(
            r"(January|February|March|April|May|June|July|August|September|October|November|December|Spring|Summer|Fall|Winter)",
            re.I,
        )
        for i, s in enumerate(texts[:-1]):
            txt = (s.get("text") or "").strip()
            if re.search(r"treasury\s+bulletin", txt, re.I):
                nxt = texts[i + 1]
                ntxt = (nxt.get("text") or "").strip()
                if date_pat.search(ntxt):
                    out.append(nxt)
        return out
    except Exception:
        return []
