def rule_page1_near_zip_code_and_ein(doc: dict) -> list[dict]:
    """Match spans near zip code and EIN markers in the page-1 cover block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'\(Zip Code\)|Employer Identification No|I\.R\.S\.', txt, re.I):
                for j in range(max(0, i - 4), min(len(texts), i + 2)):
                    out.append(texts[j])
        return out
    except Exception:
        return []
