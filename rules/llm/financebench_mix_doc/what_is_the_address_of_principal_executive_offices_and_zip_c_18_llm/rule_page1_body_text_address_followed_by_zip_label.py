def rule_page1_body_text_address_followed_by_zip_label(doc: dict) -> list[dict]:
    """Match page-1 body text spans where a nearby next span is the ZIP code label."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 2):
            a, b, c = texts[i], texts[i+1], texts[i+2]
            if not all(s.get("page_no") == 1 for s in (a, b, c)):
                continue
            ta = (a.get("text") or "").strip()
            combo = " ".join([(b.get("text") or ""), (c.get("text") or "")])
            if re.search(r'^\d{1,5}\s', ta) and re.search(r'zip code', combo, re.I):
                out.append(a)
        return out
    except Exception:
        return []
