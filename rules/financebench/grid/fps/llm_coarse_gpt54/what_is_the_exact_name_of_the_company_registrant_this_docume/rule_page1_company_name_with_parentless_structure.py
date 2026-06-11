def rule_page1_company_name_with_parentless_structure(doc: dict) -> list[dict]:
    """Match parentless page-1 company-like cover spans, which are often the registrant name."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        pat = re.compile(r'\b(inc\.?|incorporated|corporation|company|plc|co\.)\b', re.I)
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("structure", {}).get("parent_id", "x") is not None:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            if pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
