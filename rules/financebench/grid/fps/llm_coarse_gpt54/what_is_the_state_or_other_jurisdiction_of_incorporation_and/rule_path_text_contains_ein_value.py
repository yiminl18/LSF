def rule_path_text_contains_ein_value(doc: dict) -> list[dict]:
    """Match spans whose path_text contains an EIN-like value under the company block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if re.search(r"\|\s*\d{2}-\d{7}\s*$", path):
                out.append(span)
        return out
    except Exception:
        return []
