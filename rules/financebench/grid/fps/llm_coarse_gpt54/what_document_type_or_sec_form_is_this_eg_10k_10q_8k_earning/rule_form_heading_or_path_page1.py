def rule_form_heading_or_path_page1(doc: dict) -> list[dict]:
    """Match page-1 spans where either text or path_text contains a common SEC form."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "")
            path = (span.get("structure", {}).get("path_text") or "")
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K|20-F|6-K|S-1|S-3|S-4)\b", text + " " + path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
