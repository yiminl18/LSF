def rule_ffo1_summary_table(doc: dict) -> list[dict]:
    """Match table spans for FFO-1 / Summary of Fiscal Operations."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                path = ((span.get("structure") or {}).get("path_text") or "")
                if (
                    re.search(r'\bFFO[-\s]?1\b', txt, re.I)
                    or re.search(r'summary of fiscal operations', txt, re.I)
                    or re.search(r'\bFFO[-\s]?1\b', path, re.I)
                    or re.search(r'summary of fiscal operations', path, re.I)
                ):
                    out.append(span)
    except Exception:
        return []
    return out
