def rule_form_h1_exact(doc: dict) -> list[dict]:
    """Match H1/section-header spans on page 1 whose text is an SEC form like FORM 10-K, FORM 10-Q, or FORM 8-K."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and str(span.get("structure", {}).get("level", "")) == "H1"
                and re.fullmatch(r"FORM\s+(10-K|10-Q|8-K|20-F|6-K|S-1|S-3|S-4|DEF 14A|SC 13D|SC 13G)", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
