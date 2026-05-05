def rule_none_listed_when_no_exhibit_index(doc: dict) -> list[dict]:
    """Heuristic: return spans indicating no exhibit index context in documents like 10-Qs where no material agreement is listed."""
    try:
        texts = doc.get("texts", [])
        has_exhibit_index = any(
            "exhibit index" in (((s.get("text") or "") + " " + (((s.get("structure") or {}).get("path_text")) or "")).lower())
            for s in texts
        )
        if has_exhibit_index:
            return []
        return [s for s in texts if s.get("label") in {"section_header", "table"} and (
            "item 6" in (((s.get("text") or "") + " " + (((s.get("structure") or {}).get("path_text")) or "")).lower())
            or "item 15" in (((s.get("text") or "") + " " + (((s.get("structure") or {}).get("path_text")) or "")).lower())
        )]
    except Exception:
        return []
