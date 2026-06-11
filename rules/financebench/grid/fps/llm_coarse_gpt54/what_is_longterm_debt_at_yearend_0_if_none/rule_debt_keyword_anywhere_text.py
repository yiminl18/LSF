def rule_debt_keyword_anywhere_text(doc: dict) -> list[dict]:
    """Match any text span containing long-term debt or principal amount of total debt."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") in {"text", "section_header"}:
                txt = (span.get("text") or "")
                if re.search(r"\blong[- ]term debt\b|\bprincipal amount of total debt\b|\btotal debt\b", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
