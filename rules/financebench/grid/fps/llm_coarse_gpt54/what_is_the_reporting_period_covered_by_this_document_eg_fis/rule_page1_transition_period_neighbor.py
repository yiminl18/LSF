def rule_page1_transition_period_neighbor(doc: dict) -> list[dict]:
    """Match page-1 spans adjacent to transition-period lines, since the true answer often sits just above them."""
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "transition period" in text:
                for j in [i - 2, i - 1, i, i + 1]:
                    if 0 <= j < len(texts):
                        cand = texts[j]
                        ct = (cand.get("text") or "").lower()
                        if cand.get("page_no") == 1 and any(k in ct for k in [
                            "fiscal year ended", "quarterly period ended", "date of report"
                        ]):
                            out.append(cand)
        return out
    except Exception:
        return []
