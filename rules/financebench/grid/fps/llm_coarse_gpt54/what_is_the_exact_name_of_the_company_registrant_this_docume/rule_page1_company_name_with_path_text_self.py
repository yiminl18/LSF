def rule_page1_company_name_with_path_text_self(doc: dict) -> list[dict]:
    """Match page-1 spans where path_text is the same as text and text looks like a company name."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        pat = re.compile(r'\b(inc\.?|incorporated|corporation|company|plc|co\.)\b', re.I)
        for span in texts:
            txt = (span.get("text", "") or "").strip()
            path = (span.get("structure", {}).get("path_text", "") or "").strip()
            if span.get("page_no") == 1 and txt and txt == path and pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
