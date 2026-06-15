def rule_narrative_office_location(doc: dict) -> list[dict]:
    """Match narrative sentences that explicitly state where the executive offices are located."""
    try:
        patterns = (
            "principal executive offices are located at",
            "mailing address and executive offices are located at",
            "executive offices and principal facilities are located at",
            "executive offices are located at",
            "principal offices are located at",
        )

        return [
            span for span in doc.get("texts", [])
            if any(p in ((span.get("text") or "").lower()) for p in patterns)
        ]
    except Exception:
        return []
