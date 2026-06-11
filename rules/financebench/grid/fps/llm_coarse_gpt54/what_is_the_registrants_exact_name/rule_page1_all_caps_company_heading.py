def rule_page1_all_caps_company_heading(doc: dict) -> list[dict]:
    """Match all-caps prominent page-1 headings that look like company names."""
    try:
        texts = doc.get("texts", [])
        out = []
        import re
        for span in texts:
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if float(span.get("size") or 0) < 10:
                continue
            if "securities and exchange commission" in low or "form 10-" in low or "form 8-k" in low:
                continue
            letters = re.sub(r"[^A-Za-z]", "", txt)
            if not letters:
                continue
            if letters.upper() == letters and len(letters) >= 4:
                out.append(span)
        return out
    except Exception:
        return []
