def rule_page1_cover_page_candidate_cluster(doc: dict) -> list[dict]:
    """Match clusters of page-1 spans around any EIN-like number."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") == 1 and re.fullmatch(r"\d{2}-\d{7}", txt):
                for cand in texts[max(0, i-3):min(len(texts), i+4)]:
                    if cand.get("page_no") == 1:
                        out.append(cand)
        return out
    except Exception:
        return []
