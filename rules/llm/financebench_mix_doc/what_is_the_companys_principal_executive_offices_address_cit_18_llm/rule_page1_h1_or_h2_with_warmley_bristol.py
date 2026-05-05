def rule_page1_h1_or_h2_with_warmley_bristol(doc: dict) -> list[dict]:
    """Match page 1 headers containing Warmley/Bristol/United Kingdom address text."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") == "section_header":
                txt = (span.get("text") or "").lower()
                if "warmley" in txt or "bristol" in txt or "united kingdom" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
