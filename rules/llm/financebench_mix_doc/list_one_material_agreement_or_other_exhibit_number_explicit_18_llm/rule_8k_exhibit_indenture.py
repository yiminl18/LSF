def rule_8k_exhibit_indenture(doc: dict) -> list[dict]:
    """Match spans mentioning supplemental indenture exhibits, especially 4.6/4.7."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"\b4\.[67]\b", txt) and re.search(r"supplemental indenture", txt, re.I):
                out.append(span)
            elif re.search(r"supplemental indenture", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
