def rule_form_type_from_path_or_text_page1(doc: dict) -> list[dict]:
    """Match page-1 spans where either path_text or text indicates the document type."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", path, re.I) or re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", text, re.I) or "CURRENT REPORT" in text.upper():
                out.append(span)
        return out
    except Exception:
        return []
