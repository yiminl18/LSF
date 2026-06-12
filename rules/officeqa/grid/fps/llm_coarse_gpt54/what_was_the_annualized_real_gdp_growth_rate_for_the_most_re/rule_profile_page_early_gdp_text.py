def rule_profile_page_early_gdp_text(doc: dict) -> list[dict]:
    """Match early-page Profile of the Economy prose about GDP growth, where answers appear in later vintages."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if page is not None and page <= 12 and "Profile of the Economy" in path:
                if re.search(r"real\s+GDP|gross\s+domestic\s+product", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
