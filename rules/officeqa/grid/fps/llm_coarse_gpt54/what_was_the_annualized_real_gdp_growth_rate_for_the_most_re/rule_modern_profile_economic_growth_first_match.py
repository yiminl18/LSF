def rule_modern_profile_economic_growth_first_match(doc: dict) -> list[dict]:
    """Match the first modern Profile of the Economy growth paragraph mentioning the latest quarter."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if "Profile of the Economy" in path and re.search(
                r"(real GDP|gross domestic product).*?(first|second|third|fourth)\s+quarter.*?\d+\.\d+\s*percent",
                txt,
                re.I,
            ):
                out.append(span)
                break
        return out
    except Exception:
        return []
