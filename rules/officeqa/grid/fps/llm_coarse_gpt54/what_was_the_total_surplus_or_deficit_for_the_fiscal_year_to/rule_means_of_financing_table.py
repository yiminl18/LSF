def rule_means_of_financing_table(doc: dict) -> list[dict]:
    """Match tables combining surplus/deficit with means of financing."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'means of financing', txt, re.I) and re.search(r'(surplus|deficit)', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
