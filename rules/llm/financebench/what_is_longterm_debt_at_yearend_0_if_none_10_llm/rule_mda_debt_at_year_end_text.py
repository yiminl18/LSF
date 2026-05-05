def rule_mda_debt_at_year_end_text(doc: dict) -> list[dict]:
    """Match text spans that explicitly say debt at year-end or at fiscal year-end."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "section_header"}:
                continue
            txt = (span.get("text") or "").lower()
            if (
                ("year-end" in txt or "year end" in txt or "fiscal year-end" in txt or "at december" in txt or "at june" in txt)
                and (re.search(r"\blong[\-\s]?term debt\b", txt) or "debt" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
