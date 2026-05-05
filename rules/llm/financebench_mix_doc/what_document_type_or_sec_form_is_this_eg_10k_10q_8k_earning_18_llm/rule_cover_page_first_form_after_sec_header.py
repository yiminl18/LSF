def rule_cover_page_first_form_after_sec_header(doc: dict) -> list[dict]:
    """Match the first form heading appearing after the SEC commission header on page 1."""
    import re
    try:
        texts = doc.get("texts", [])
        sec_seen = False
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").upper()
            if "SECURITIES AND EXCHANGE COMMISSION" in txt:
                sec_seen = True
                continue
            if sec_seen and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", txt, re.I):
                out.append(span)
                break
        return out
    except Exception:
        return []
