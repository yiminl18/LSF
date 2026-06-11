def rule_form_code_breadcrumb_exact(doc: dict) -> list[dict]:
    """Match spans whose breadcrumb path_text is exactly a common form code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            p = ((span.get("structure") or {}).get("path_text") or "").strip()
            if re.fullmatch(r"FORM\s+(10-K|10-Q|8-K)", p, re.I):
                out.append(span)
        return out
    except Exception:
        return []
