def rule_top_pages_bold_non_generic_subject(doc: dict) -> list[dict]:
    """Match bold non-generic spans on pages 1-3 that look like legal subject labels in the summary area."""
    try:
        out = []
        bad = {"SUMMARY", "COUNSEL", "OPINION", "ORDER", "BACKGROUND"}
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            if span.get("page_no") in {1, 2, 3} and span.get("bold") == 1 and txt:
                if up in bad or any(up.startswith(b) for b in bad):
                    continue
                if span.get("label") in {"section_header", "text"}:
                    out.append(span)
        return out
    except Exception:
        return []
