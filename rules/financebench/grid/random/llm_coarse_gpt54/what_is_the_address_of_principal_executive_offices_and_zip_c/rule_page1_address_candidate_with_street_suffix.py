def rule_page1_address_candidate_with_street_suffix(doc: dict) -> list[dict]:
    """Match page-1 spans containing common street suffixes."""
    try:
        import re
        suffixes = r'\b(?:Street|St\.|Avenue|Ave\.|Boulevard|Blvd\.|Drive|Dr\.|Road|Rd\.|Center|Plaza|Building)\b'
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1 and re.search(suffixes, (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
