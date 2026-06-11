def rule_page1_h1_company_like_name(doc: dict) -> list[dict]:
    """Match page-1 H1 headers whose text looks like a company name by suffix or punctuation."""
    try:
        import re
        out = []
        suffix_re = re.compile(r"\b(inc\.?|corporation|company|plc|ltd\.?|limited)\b", re.I)
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and suffix_re.search(txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
