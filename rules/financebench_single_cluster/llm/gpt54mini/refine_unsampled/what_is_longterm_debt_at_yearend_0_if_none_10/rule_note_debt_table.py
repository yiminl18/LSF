def rule_note_debt_table(doc: dict) -> list[dict]:
    """Match debt note tables under notes to consolidated financial statements that mention long-term debt."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if (
                "notes to consolidated financial statements" in path
                or re.search(r"\bnote\b", path)
                or re.search(r"\bdebt\b", path)
            ):
                if re.search(r"\blong[\-\s]?term debt\b", txt) or "debt excluding current maturities" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
