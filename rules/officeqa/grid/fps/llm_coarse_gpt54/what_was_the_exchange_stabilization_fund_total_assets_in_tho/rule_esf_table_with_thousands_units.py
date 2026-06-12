def rule_esf_table_with_thousands_units(doc: dict) -> list[dict]:
    """Match ESF-related tables that mention thousands or in thousands of dollars."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            combo = f"{path}\n{text}"
            if re.search(r'ESF|Exchange Stabilization Fund', combo, re.I) and re.search(r'thousand|thousands', combo, re.I):
                out.append(span)
        return out
    except Exception:
        return []
