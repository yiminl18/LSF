def rule_under_company_header_page1_exchange(doc: dict) -> list[dict]:
    """Match page-1 exchange spans that are under the company-name cover section."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and path and re.search(r'new york stock exchange|nasdaq|global select market', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
