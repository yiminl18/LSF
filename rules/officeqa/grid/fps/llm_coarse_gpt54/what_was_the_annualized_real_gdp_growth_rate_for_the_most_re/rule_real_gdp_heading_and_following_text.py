def rule_real_gdp_heading_and_following_text(doc: dict) -> list[dict]:
    """Match the 'Real gross domestic product' heading and the next few body spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if re.fullmatch(r"Real gross domestic product", txt, re.I):
                out.append(span)
                for j in range(i + 1, min(i + 5, len(texts))):
                    out.append(texts[j])
        return out
    except Exception:
        return []
