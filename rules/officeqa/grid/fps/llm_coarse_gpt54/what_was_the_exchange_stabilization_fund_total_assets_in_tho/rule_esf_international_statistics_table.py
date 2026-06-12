def rule_esf_international_statistics_table(doc: dict) -> list[dict]:
    """Match tables in the international statistics area that mention ESF or Exchange Stabilization Fund."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = span.get("text") or ""
            if re.search(r'INTERNATIONAL', path, re.I) and re.search(r'ESF|Exchange Stabilization Fund', path + " " + text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
