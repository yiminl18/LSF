def rule_path_contains_form_and_period(doc: dict) -> list[dict]:
    """Match spans whose path_text contains FORM and whose text contains a period phrase."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            txt = (span.get("text") or "").lower()
            if "form" in path and re.search(r'(fiscal year ended|quarterly period ended|date of report)', txt):
                out.append(span)
        return out
    except Exception:
        return []
