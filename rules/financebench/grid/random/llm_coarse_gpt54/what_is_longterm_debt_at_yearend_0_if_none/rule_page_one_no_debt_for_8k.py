def rule_page_one_no_debt_for_8k(doc: dict) -> list[dict]:
    """Return page-1 issuer summary spans for 8-Ks where the correct answer is often 0 because no long-term debt disclosure exists."""
    try:
        import re
        out = []
        has_8k = any(re.search(r"form 8-k", (s.get("text", "") or ""), re.I) for s in doc.get("texts", []))
        if not has_8k:
            return []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") in {"section_header", "text"}:
                out.append(span)
        return out
    except Exception:
        return []
