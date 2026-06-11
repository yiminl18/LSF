def rule_exchange_in_higher_level_cover_headers(doc: dict) -> list[dict]:
    """Match exchange mentions in H1/H2/H3 cover-page headers."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            lvl = (span.get("structure", {}) or {}).get("level")
            if span.get("page_no") == 1 and lvl in {"H1", "H2", "H3"}:
                txt = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
                if re.search(r'new york stock exchange|nasdaq|global select market', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
