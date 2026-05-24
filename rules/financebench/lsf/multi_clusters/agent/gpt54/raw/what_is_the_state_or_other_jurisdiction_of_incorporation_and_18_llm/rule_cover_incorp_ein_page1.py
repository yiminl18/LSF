def rule_cover_incorp_ein_page1(doc: dict) -> list[dict]:
    """Retrieve first-page cover block spans around incorporation state and IRS EIN."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for s in texts:
            if s.get("page_no") != 1:
                continue
            txt = (s.get("text") or "")
            path = ((s.get("structure") or {}).get("path_text") or "")
            label = s.get("label") or ""
            level = ((s.get("structure") or {}).get("level") or "")
            low = txt.lower()
            path_low = path.lower()
            if "state or other jurisdiction of incorporation" in low or "state or other jurisdiction of incorporation or organization" in low:
                out.append(s)
                continue
            if "i.r.s. employer identification no" in low or "irs employer identification no" in low:
                out.append(s)
                continue
            if re.search(r"\b\d{2}-\d{7}\b", txt):
                out.append(s)
                continue
            if label in {"section_header", "text"} and level in {"H1", "H2", "Body"}:
                if path and all(x not in path_low for x in ["part i", "item 1", "risk factors", "table of contents", "index"]):
                    if s not in out:
                        out.append(s)
        return out
    except Exception:
        return []

