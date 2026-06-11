def rule_page1_h1_with_company_suffix(doc: dict) -> list[dict]:
    """Match page-1 H1/header spans whose text ends with common company suffixes."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        pat = re.compile(r'\b(inc\.?|incorporated|corporation|corp\.?|company|co\.?,?\s+inc\.?|plc|ltd\.?)\b', re.I)
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            if pat.search(txt):
                if "securities and exchange commission" in txt.lower():
                    continue
                out.append(span)
        return out
    except Exception:
        return []
