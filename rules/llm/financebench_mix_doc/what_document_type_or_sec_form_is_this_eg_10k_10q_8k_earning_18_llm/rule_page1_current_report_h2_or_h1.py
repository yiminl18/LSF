def rule_page1_current_report_h2_or_h1(doc: dict) -> list[dict]:
    """Match page-1 CURRENT REPORT headings at H1/H2/body level."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            level = ((span.get("structure") or {}).get("level") or "")
            if level in {"H1", "H2", "Body"} and "CURRENT REPORT" in ((span.get("text") or "").upper()):
                out.append(span)
        return out
    except Exception:
        return []
