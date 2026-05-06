def rule_page1_h2_h3_registration_headers(doc: dict) -> list[dict]:
    """Match page 1 H2/H3 section headers that are part of the registration block."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            level = ((span.get("structure", {}) or {}).get("level") or "")
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if level in {"H2", "H3", "H4"} and (
                "trading symbol" in txt
                or "name of each exchange" in txt
                or "section 12(b)" in txt
                or "nasdaq" in txt
                or "new york stock exchange" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
