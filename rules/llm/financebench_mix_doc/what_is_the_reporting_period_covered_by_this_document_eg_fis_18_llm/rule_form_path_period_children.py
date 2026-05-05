def rule_form_path_period_children(doc: dict) -> list[dict]:
    """Match spans under FORM 10-K/10-Q/8-K whose text contains the period phrase."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "").upper()
            txt = (span.get("text") or "").lower()
            if "FORM 10-" in path or "FORM 8-K" in path:
                if re.search(r'for the (fiscal year|quarterly period|quarter|period) ended', txt) or re.search(r'date of report \(date of earliest event reported\)', txt):
                    out.append(span)
        return out
    except Exception:
        return []
