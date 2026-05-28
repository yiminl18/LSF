def rule_page2_bold_subject_heading(doc: dict) -> list[dict]:
    """Match bold heading-like spans on page 2 that are not SUMMARY/COUNSEL/OPINION and look like subject labels."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 2 and span.get("bold") == 1 and txt:
                up = txt.upper()
                if "SUMMARY" in up or "COUNSEL" in up or up == "OPINION" or up == "BACKGROUND":
                    continue
                if span.get("label") in {"section_header", "text"}:
                    out.append(span)
        return out
    except Exception:
        return []
