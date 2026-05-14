def rule_debt_table_under_notes_or_mda(doc: dict) -> list[dict]:
    """Match tables under either notes or MD&A paths that mention debt."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if (
                ("notes to consolidated financial statements" in path or "management's discussion and analysis" in path)
                and (re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt))
            ):
                out.append(span)
        return out
    except Exception:
        return []
