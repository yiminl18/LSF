def rule_international_statistics_section_near_foreign_currency(doc: dict) -> list[dict]:
    """Match spans in the international statistics area that mention foreign currency positions or Canadian dollar positions."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "international statistics" in path or "international financial statistics" in path:
                if "foreign currency positions" in txt or "canadian dollar positions" in txt or "fcp-" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
