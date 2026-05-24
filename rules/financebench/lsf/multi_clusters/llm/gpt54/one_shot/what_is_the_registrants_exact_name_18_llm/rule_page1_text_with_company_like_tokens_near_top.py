def rule_page1_text_with_company_like_tokens_near_top(doc: dict) -> list[dict]:
    """Match page-1 text or section_header spans with company suffixes near the top of the filing."""
    try:
        import re
        out = []
        pat = re.compile(r"\b(inc\.?|company|corporation|plc)\b", re.I)
        for idx, span in enumerate(doc.get("texts", [])[:40]):
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if (
                span.get("page_no") == 1
                and pat.search(txt)
                and "securities and exchange commission" not in low
                and "current report" not in low
                and "form 10-" not in low
                and "form 8-k" not in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
