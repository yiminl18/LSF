def rule_profile_of_economy_real_gdp_heading_modern(doc: dict) -> list[dict]:
    """Match modern headings and text for real GDP in Profile of the Economy."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if re.search(r"Growth of Real GDP|Economic Growth|Real gross domestic product", txt, re.I):
                path = ((span.get("structure") or {}).get("path_text") or "")
                if "Profile of the Economy" in path or span.get("page_no", 999) <= 12:
                    out.append(span)
                    for j in range(i + 1, min(i + 4, len(texts))):
                        out.append(texts[j])
        return out
    except Exception:
        return []
