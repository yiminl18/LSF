def rule_exchange_after_cover_company_name(doc: dict) -> list[dict]:
    """Match exchange spans appearing after the main company-name header on the cover page."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        company_idxs = []
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and s.get("label") in ("section_header", "text"):
                txt = (s.get("text") or "").strip()
                if txt and re.search(r'(inc\.|incorporated|corporation|company|plc|co\.)', txt, re.I):
                    if not re.search(r'commission|form 10-|current report|annual report', txt, re.I):
                        company_idxs.append(i)
        for idx in company_idxs[:3]:
            for j in range(idx, min(len(texts), idx + 25)):
                s = texts[j]
                if s.get("page_no") == 1:
                    t = (s.get("text") or "").strip()
                    if re.search(r'new york stock exchange|the nasdaq global select market|the nasdaq stock market llc|NASDAQ\b', t, re.I):
                        out.append(s)
        return out
    except Exception:
        return []
