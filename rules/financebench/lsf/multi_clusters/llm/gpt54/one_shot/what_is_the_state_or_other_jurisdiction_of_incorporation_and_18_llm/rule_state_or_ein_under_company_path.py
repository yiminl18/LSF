def rule_state_or_ein_under_company_path(doc: dict) -> list[dict]:
    """Match spans under a non-FORM company path that contain either a jurisdiction value or EIN pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if path and "FORM 10-" not in path.upper() and (
                re.search(r"\b\d{2}-\d{7}\b", text)
                or re.search(r"\b(Delaware|Washington|New York|Jersey)\b", text, re.I)
                or re.search(r"state or other jurisdiction|employer identification", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
