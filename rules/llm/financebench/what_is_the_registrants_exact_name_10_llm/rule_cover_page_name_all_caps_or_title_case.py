def rule_cover_page_name_all_caps_or_title_case(doc: dict) -> list[dict]:
    """Match page-1 company-name candidates in all caps or title case near the exact-name area."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if not txt or len(txt) < 3:
                continue
            low = txt.lower()
            if "form 10-k" in low or "commission file" in low or "annual report" in low:
                continue
            if "exact name of registrant" in low:
                continue
            alpha = "".join(ch for ch in txt if ch.isalpha())
            if not alpha:
                continue
            if txt == txt.upper() or txt[:1].isupper():
                if span.get("bold") == 1:
                    out.append(span)
        return out
    except Exception:
        return []
