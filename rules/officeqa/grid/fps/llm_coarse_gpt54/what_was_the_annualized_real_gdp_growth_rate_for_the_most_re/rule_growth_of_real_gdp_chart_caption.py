def rule_growth_of_real_gdp_chart_caption(doc: dict) -> list[dict]:
    """Match chart captions/titles for Growth of Real GDP and nearby explanatory text."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if re.search(r"Growth of Real GDP", txt, re.I):
                out.append(span)
                for j in range(max(0, i - 1), min(i + 4, len(texts))):
                    out.append(texts[j])
        return out
    except Exception:
        return []
