def rule_profile_of_economy_gdp_chart_and_text(doc: dict) -> list[dict]:
    """Match GDP chart title/caption plus adjacent explanatory text in the economy profile."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if re.search(r"Growth of Real GDP", txt, re.I):
                for j in range(max(0, i - 2), min(len(texts), i + 6)):
                    out.append(texts[j])
        return out
    except Exception:
        return []
