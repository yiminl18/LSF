def rule_path_text_contains_state_value(doc: dict) -> list[dict]:
    """Match spans whose path_text itself contains a likely state/jurisdiction value under the company block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if re.search(r"\|\s*(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)\s*$", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
