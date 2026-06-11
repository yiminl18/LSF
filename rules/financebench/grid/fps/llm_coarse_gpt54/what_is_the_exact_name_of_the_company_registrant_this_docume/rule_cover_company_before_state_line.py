def rule_cover_company_before_state_line(doc: dict) -> list[dict]:
    """Match a company-like span immediately before a state/jurisdiction line on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a = texts[i]
            b = texts[i + 1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            bblob = ((b.get("text", "") or "") + " " + (b.get("text_span", "") or "")).lower()
            if "state or other jurisdiction of incorporation" in bblob:
                atxt = (a.get("text", "") or "").strip()
                if atxt and "form 10-" not in atxt.lower() and "form 8-k" not in atxt.lower():
                    out.append(a)
        return out
    except Exception:
        return []
