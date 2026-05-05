def rule_form_heading_any_page(doc: dict) -> list[dict]:
    """Match any span whose text contains a canonical SEC form heading."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", (span.get("text") or "").strip(), re.I)
        ]
    except Exception:
        return []
