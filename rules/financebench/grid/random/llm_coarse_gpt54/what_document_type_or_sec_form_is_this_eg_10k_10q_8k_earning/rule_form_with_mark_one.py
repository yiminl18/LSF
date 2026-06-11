def rule_form_with_mark_one(doc: dict) -> list[dict]:
    """Match spans containing a form code and '(Mark One)', a common cover-page cue."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I)
            and "MARK ONE" in (span.get("text") or "").upper()
        ]
    except Exception:
        return []
