def rule_cover_page_form_near_sec_header(doc: dict) -> list[dict]:
    """Match form spans on page 1 appearing after SEC/Commission header text."""
    import re
    try:
        texts = doc.get("texts", [])
        sec_seen = False
        out = []
        for span in texts[:25]:
            t = (span.get("text") or "").upper()
            if span.get("page_no") == 1 and ("SECURITIES AND EXCHANGE COMMISSION" in t or "WASHINGTON, D.C. 20549" in t):
                sec_seen = True
            if sec_seen and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
