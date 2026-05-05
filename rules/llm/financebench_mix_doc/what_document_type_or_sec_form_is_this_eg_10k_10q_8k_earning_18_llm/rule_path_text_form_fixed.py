def rule_path_text_form_fixed(doc: dict) -> list[dict]:
    """Match spans whose breadcrumb path_text contains a form name."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
