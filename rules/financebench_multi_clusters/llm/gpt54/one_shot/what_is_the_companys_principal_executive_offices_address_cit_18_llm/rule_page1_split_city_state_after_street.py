def rule_page1_split_city_state_after_street(doc: dict) -> list[dict]:
    """Match split city/state spans that follow a street-address span on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            txt = span.get("text") or ""
            if span.get("page_no") != 1:
                continue
            if re.search(r"^\d{1,5}\s", txt):
                for j in range(i + 1, min(len(texts), i + 4)):
                    cand = texts[j]
                    ctext = cand.get("text") or ""
                    if cand.get("page_no") == 1 and re.search(r"\b[A-Z][a-zA-Z\.\- ]+,?$", ctext):
                        out.append(cand)
                    if cand.get("page_no") == 1 and re.fullmatch(r"[A-Z]{2}", ctext.strip()):
                        out.append(cand)
        return out
    except Exception:
        return []
