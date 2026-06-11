def rule_cover_company_before_exact_name_line(doc: dict) -> list[dict]:
    """Match a company-like span immediately before the exact-name annotation line on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a = texts[i]
            b = texts[i + 1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            if "exact name of registrant as specified in its charter" in (b.get("text", "") or "").lower():
                out.append(a)
        return out
    except Exception:
        return []
