def rule_page1_costco_issaquah_header(doc: dict) -> list[dict]:
    """Match Costco-style page-1 address line with Issaquah, WA."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'999 Lake Drive.*Issaquah,\s*WA\s*98027', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
