def rule_recent_template_profile_pages_7_to_11(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans on pages 7-11 in recent Treasury Bulletin templates."""
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if page is not None and 7 <= page <= 11 and "profile of the economy" in path.lower():
                out.append(span)
        return out
    except Exception:
        return []
