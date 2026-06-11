def rule_page1_h1_with_company_suffix(doc: dict) -> list[dict]:
    """Match page-1 H1 headers ending with common company suffixes like Inc., plc, Corporation, Company."""
    try:
        import re
        out = []
        pat = re.compile(r"\b(inc\.?|plc|corporation|company|incorporated)\b", re.I)
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and pat.search(txt)
                and "FORM 10-" not in txt.upper()
            ):
                out.append(span)
        return out
    except Exception:
        return []
