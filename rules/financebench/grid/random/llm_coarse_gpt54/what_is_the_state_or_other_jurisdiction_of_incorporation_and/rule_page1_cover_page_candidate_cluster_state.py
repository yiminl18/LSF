def rule_page1_cover_page_candidate_cluster_state(doc: dict) -> list[dict]:
    """Match clusters of page-1 spans around any likely state/jurisdiction value."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") == 1 and re.fullmatch(r"Delaware|Washington|New York|Jersey(?: \(Channel Islands\))?", txt, re.I):
                for cand in texts[max(0, i-3):min(len(texts), i+4)]:
                    if cand.get("page_no") == 1:
                        out.append(cand)
        return out
    except Exception:
        return []
