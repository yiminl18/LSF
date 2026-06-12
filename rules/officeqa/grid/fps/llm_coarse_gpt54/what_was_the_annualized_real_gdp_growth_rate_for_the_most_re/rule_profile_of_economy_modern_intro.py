def rule_profile_of_economy_modern_intro(doc: dict) -> list[dict]:
    """Match modern Profile of the Economy intro/growth spans that usually contain the latest quarter GDP rate."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if re.fullmatch(r"Introduction|Economic Growth|Growth", txt.strip(), re.I):
                path = ((span.get("structure") or {}).get("path_text") or "")
                if "Profile of the Economy" in path:
                    out.append(span)
                    for j in range(i + 1, min(i + 5, len(texts))):
                        s = texts[j]
                        if "Profile of the Economy" in (((s.get("structure") or {}).get("path_text") or "")):
                            out.append(s)
        return out
    except Exception:
        return []
