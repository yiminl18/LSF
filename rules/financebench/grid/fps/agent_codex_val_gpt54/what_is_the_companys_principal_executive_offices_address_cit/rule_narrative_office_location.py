def rule_narrative_office_location(doc: dict) -> list[dict]:
    """Match later narrative sentences that restate the office or headquarters location."""
    try:
        patterns = (
            "principal executive offices are located at",
            "mailing address and executive offices are located at",
            "executive offices and principal facilities are located at",
            "executive offices are located at",
            "principal corporate and administrative offices",
            "corporate headquarters are located",
            "headquartered in ",
        )

        return [
            span for span in doc.get("texts", [])
            if any(p in (span.get("text") or "").lower() for p in patterns)
        ]
    except Exception:
        return []
