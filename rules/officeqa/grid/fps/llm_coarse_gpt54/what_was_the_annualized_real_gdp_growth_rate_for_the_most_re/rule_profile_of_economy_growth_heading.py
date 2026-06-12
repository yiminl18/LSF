def rule_profile_of_economy_growth_heading(doc: dict) -> list[dict]:
    """Match the Growth/Economic Growth/Real gross domestic product subsection and nearby text."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        trigger_idxs = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Profile of the Economy" in path and re.search(
                r"^(Growth|Economic Growth|Real gross domestic product|Growth of Real GDP)$",
                txt.strip(),
                re.I,
            ):
                trigger_idxs.append(i)
                out.append(span)
        for i in trigger_idxs:
            for j in range(i + 1, min(i + 6, len(texts))):
                s = texts[j]
                if ((s.get("structure") or {}).get("path_text") or "").find("Profile of the Economy") >= 0:
                    out.append(s)
        return out
    except Exception:
        return []
