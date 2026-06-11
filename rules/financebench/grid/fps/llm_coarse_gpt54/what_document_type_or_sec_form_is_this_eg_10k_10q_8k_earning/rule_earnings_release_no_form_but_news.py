def rule_earnings_release_no_form_but_news(doc: dict) -> list[dict]:
    """Match earnings-release documents that may not be SEC-form-first but start with News Release and financial-results language."""
    import re
    try:
        texts = doc.get("texts", [])
        has_news = any((s.get("text") or "").strip().lower() == "news release" for s in texts if s.get("page_no") == 1)
        if not has_news:
            return []
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r"(news release|financial results|reaffirms .* outlook|net sales of)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
