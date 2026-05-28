def rule_summary_heading_immediately_after_summary(doc: dict) -> list[dict]:
    """Match a bold summary subject section header that appears immediately after a SUMMARY marker on page 2."""
    try:
        import re
        out = []
        texts = doc.get("texts", [])
        generic = {"SUMMARY", "COUNSEL", "OPINION", "ORDER", "BACKGROUND", "FILED"}
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            if span.get("page_no") != 2 or span.get("label") != "section_header" or span.get("bold") != 1 or not txt:
                continue
            if up in generic:
                continue
            j = i - 1
            while j >= 0 and not (texts[j].get("text") or "").strip():
                j -= 1
            if j >= 0 and re.search(r"(^|\W)SUMMARY(\W|$)", (texts[j].get("text") or "").upper()):
                out.append(span)
        return out
    except Exception:
        return []
