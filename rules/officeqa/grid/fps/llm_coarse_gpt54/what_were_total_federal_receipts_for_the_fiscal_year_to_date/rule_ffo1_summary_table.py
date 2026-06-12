def rule_ffo1_summary_table(doc: dict) -> list[dict]:
    """Match table spans for Table FFO-1 / Summary of Fiscal Operations."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                path = ((span.get("structure") or {}).get("path_text") or "")
                if (
                    re.search(r'FFO[-\s]?1', txt, re.I)
                    or re.search(r'Summary of Fiscal Operations', txt, re.I)
                    or re.search(r'FFO[-\s]?1', path, re.I)
                    or re.search(r'Summary of Fiscal Operations', path, re.I)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
