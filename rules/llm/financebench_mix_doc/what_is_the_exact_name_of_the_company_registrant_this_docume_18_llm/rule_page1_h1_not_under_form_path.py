def rule_page1_h1_not_under_form_path(doc: dict) -> list[dict]:
    """Match top-level page-1 H1 spans whose path_text is just the company name, not a form section path."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = span.get("structure", {}).get("path_text") or ""
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and path == txt
                and "FORM" not in path.upper()
                and "CURRENT REPORT" not in path.upper()
                and "COMMISSION" not in path.upper()
            ):
                out.append(span)
        return out
    except Exception:
        return []
