def rule_page1_amcor_warmley_bristol_header(doc: dict) -> list[dict]:
    """Match Amcor-style page-1 address line with Warmley, Bristol, United Kingdom."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'83 Tower Road North.*Warmley,\s*Bristol.*United Kingdom', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
