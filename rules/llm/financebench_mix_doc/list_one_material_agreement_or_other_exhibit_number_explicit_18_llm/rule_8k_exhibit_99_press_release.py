def rule_8k_exhibit_99_press_release(doc: dict) -> list[dict]:
    """Match spans mentioning Exhibit 99.1 press release."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"\b99\.1\b", txt) and re.search(r"press release", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
