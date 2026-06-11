def rule_form_10q_quarterly_cluster(doc: dict) -> list[dict]:
    """Match the cluster of page-1 spans around FORM 10-Q and quarterly-report language."""
    import re
    try:
        texts = doc.get("texts", [])
        has_10q = any(re.fullmatch(r"FORM\s+10-Q", (s.get("text") or "").strip(), re.I) for s in texts if s.get("page_no") == 1)
        if not has_10q:
            return []
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r"(FORM\s+10-Q|QUARTERLY REPORT|For the quarterly period ended)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
