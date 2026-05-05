def rule_8k_cover_date_body_lines(doc: dict) -> list[dict]:
    """Match body lines on the 8-K cover page containing the event date."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "")
            lvl = (span.get("structure", {}) or {}).get("level", "")
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and lvl == "Body" and ("FORM 8-K" in path or "CURRENT REPORT" in path or "FORM 8-K" in (span.get("text_span") or "")):
                if "date of report (date of earliest event reported)" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
