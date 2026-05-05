def rule_page1_first_large_formish_header(doc: dict) -> list[dict]:
    """Match the first large bold page-1 header that looks like a form/report type."""
    import re
    try:
        candidates = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            txt = (s.get("text") or "").strip()
            if float(s.get("size") or 0) >= 10 and s.get("bold") == 1:
                if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", txt, re.I) or "CURRENT REPORT" in txt.upper():
                    candidates.append(s)
        return candidates[:1]
    except Exception:
        return []
