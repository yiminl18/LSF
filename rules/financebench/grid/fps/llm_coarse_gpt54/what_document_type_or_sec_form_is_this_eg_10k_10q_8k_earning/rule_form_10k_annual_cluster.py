def rule_form_10k_annual_cluster(doc: dict) -> list[dict]:
    """Match the cluster of page-1 spans around FORM 10-K and annual-report language."""
    import re
    try:
        texts = doc.get("texts", [])
        has_10k = any(re.fullmatch(r"FORM\s+10-K", (s.get("text") or "").strip(), re.I) for s in texts if s.get("page_no") == 1)
        if not has_10k:
            return []
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r"(FORM\s+10-K|ANNUAL REPORT|For the fiscal year ended)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
