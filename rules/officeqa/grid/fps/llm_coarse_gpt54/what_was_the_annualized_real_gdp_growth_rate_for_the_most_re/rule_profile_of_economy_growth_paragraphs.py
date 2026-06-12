def rule_profile_of_economy_growth_paragraphs(doc: dict) -> list[dict]:
    """Match all body paragraphs under Growth/Economic Growth/Real GDP subsections."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        active = False
        for span in texts:
            txt = (span.get("text") or "").strip()
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Profile of the Economy" in path and re.fullmatch(r"Growth|Economic Growth|Real gross domestic product|Growth of Real GDP", txt, re.I):
                active = True
                out.append(span)
                continue
            if active:
                if span.get("label") == "section_header" and txt and not re.fullmatch(r"Growth|Economic Growth|Real gross domestic product|Growth of Real GDP", txt, re.I):
                    active = False
                else:
                    out.append(span)
        return out
    except Exception:
        return []
