def rule_page1_city_state_split_under_ein(doc: dict) -> list[dict]:
    """Match city/state fragments that appear after EIN-related headers on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").lower()
            tsp = (span.get("text_span") or "").lower()
            if "employer identification no" in txt or "employer identification no" in tsp:
                for j in range(i + 1, min(len(texts), i + 5)):
                    cand = texts[j]
                    ctext = cand.get("text") or ""
                    if cand.get("page_no") == 1 and (re.search(r"[A-Z][a-zA-Z ]+,?$", ctext) or re.fullmatch(r"[A-Z]{2}", ctext.strip())):
                        out.append(cand)
        return out
    except Exception:
        return []
