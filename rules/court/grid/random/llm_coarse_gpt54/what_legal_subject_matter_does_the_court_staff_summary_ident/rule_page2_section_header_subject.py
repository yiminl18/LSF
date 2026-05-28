def rule_page2_section_header_subject(doc: dict) -> list[dict]:
    """Match section_header spans on page 2 that likely serve as the court-staff summary subject heading."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            if span.get("page_no") == 2 and span.get("label") == "section_header" and txt:
                if "SUMMARY" in up or "COUNSEL" in up or up == "OPINION" or up == "BACKGROUND":
                    continue
                out.append(span)
        return out
    except Exception:
        return []
