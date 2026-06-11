def rule_page1_cover_address_after_state_until_ein(doc: dict) -> list[dict]:
    """Match spans in the page-1 cover block after state/incorporation label until EIN marker."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        started = False
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'State or other jurisdiction of incorporation|State or other jurisdiction of incorporation or organization', txt, re.I):
                started = True
                continue
            if started:
                if re.search(r'Employer Identification No|I\.R\.S\.', txt, re.I):
                    break
                out.append(span)
        return out
    except Exception:
        return []
