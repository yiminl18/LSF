def rule_page1_title_case_company_heading(doc: dict) -> list[dict]:
    """Match title-case prominent page-1 headings that are followed by exact-name caption text."""
    try:
        texts = doc.get("texts", [])
        out = []
        import re
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if float(span.get("size") or 0) < 12:
                continue
            if not re.search(r"[A-Z][a-z]", txt):
                continue
            if any("exact name of registrant" in ((s.get("text") or "").lower()) for s in texts[i:i+4]):
                out.append(span)
        return out
    except Exception:
        return []
