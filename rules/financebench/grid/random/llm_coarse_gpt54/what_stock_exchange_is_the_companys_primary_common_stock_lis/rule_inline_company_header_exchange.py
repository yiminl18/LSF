def rule_inline_company_header_exchange(doc: dict) -> list[dict]:
    """Match company header spans whose long text_span inline content contains exchange names."""
    try:
        import re
        texts = doc.get("texts", [])
        pats = [
            r"\bnew york stock exchange\b",
            r"\bthe new york stock exchange\b",
            r"\bnasdaq\b",
            r"\bnasdaq global select market\b",
            r"\bthe nasdaq global select market\b",
        ]
        out = []
        for span in texts:
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = ((span.get("text_span") or "") + " " + (span.get("text") or "")).lower()
                if any(re.search(p, txt) for p in pats):
                    out.append(span)
        return out
    except Exception:
        return []
