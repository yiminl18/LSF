def rule_page1_near_company_name_and_section12b(doc: dict) -> list[dict]:
    """Match page 1 spans under company-name path_text where Section 12(b) registration data is usually listed."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure", {}) or {}).get("path_text") or "").lower()
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if path and "form 10-k" not in path and (
                "section 12(b)" in txt
                or "trading symbol" in txt
                or "name of each exchange" in txt
                or "nasdaq" in txt
                or "new york stock exchange" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
