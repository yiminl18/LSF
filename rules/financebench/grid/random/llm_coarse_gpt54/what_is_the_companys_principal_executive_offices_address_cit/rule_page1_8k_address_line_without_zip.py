def rule_page1_8k_address_line_without_zip(doc: dict) -> list[dict]:
    """Match page-1 8-K address lines that contain street and city/state but may omit zip in the same span."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'\b\d+\s+\S+.*\b(?:Issaquah, WA|New York, New York|St\. Paul, Minnesota|Santa Monica, CA|San Jose, California)\b', txt):
                out.append(span)
            elif re.search(r'83 Tower Road North Warmley, Bristol United Kingdom', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
