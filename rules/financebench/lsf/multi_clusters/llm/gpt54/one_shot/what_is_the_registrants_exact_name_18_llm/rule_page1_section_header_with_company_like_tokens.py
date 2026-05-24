def rule_page1_section_header_with_company_like_tokens(doc: dict) -> list[dict]:
    """Match page-1 section headers containing company suffixes like inc, company, corporation, plc."""
    try:
        import re
        out = []
        pat = re.compile(r"\b(inc\.?|company|corporation|plc)\b", re.I)
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and pat.search(txt)
                and "exact name of registrant" not in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
