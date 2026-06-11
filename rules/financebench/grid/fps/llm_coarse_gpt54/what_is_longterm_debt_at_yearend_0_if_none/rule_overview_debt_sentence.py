def rule_overview_debt_sentence(doc: dict) -> list[dict]:
    """Match overview sentences summarizing debt reduction or debt balance at year-end."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            if "overview" in path:
                if re.search(r"\breduced our debt\b", txt, re.I) or re.search(r"\btotal debt\b.*\bcompared to\b", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
