def rule_mda_liquidity_debt_paragraph(doc: dict) -> list[dict]:
    """Match MD&A liquidity/capital resources text spans mentioning long-term debt at year-end."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "section_header"}:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if (
                "management's discussion and analysis" in path
                or "liquidity" in path
                or "capital resources" in path
            ):
                if re.search(r"\blong[\-\s]?term debt\b", txt) or re.search(r"\bdebt\b", txt):
                    out.append(span)
        return out
    except Exception:
        return []
