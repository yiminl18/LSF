def rule_profile_of_economy_quarterly_percent_change_caption(doc: dict) -> list[dict]:
    """Match captions saying '(Quarterly percent change at annual rate)' near GDP growth charts."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if re.search(r"Quarterly percent change at annual rate", txt, re.I):
                nearby = texts[max(0, i - 2): min(len(texts), i + 3)]
                if any(re.search(r"Growth of Real GDP|real gross domestic product|GDP", (s.get("text") or ""), re.I) for s in nearby):
                    out.extend(nearby)
        return out
    except Exception:
        return []
