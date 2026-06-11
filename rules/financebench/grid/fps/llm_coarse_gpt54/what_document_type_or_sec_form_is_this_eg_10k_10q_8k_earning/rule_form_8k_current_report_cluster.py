def rule_form_8k_current_report_cluster(doc: dict) -> list[dict]:
    """Match the cluster of page-1 spans around FORM 8-K and CURRENT REPORT."""
    import re
    try:
        texts = doc.get("texts", [])
        has_8k = any(re.fullmatch(r"FORM\s+8-K", (s.get("text") or "").strip(), re.I) for s in texts if s.get("page_no") == 1)
        if not has_8k:
            return []
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r"(FORM\s+8-K|CURRENT REPORT|Date of Report)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
