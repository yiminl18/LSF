def rule_item1_business_principal_offices_sentence(doc: dict) -> list[dict]:
    """Match Item 1 business sentences that state the principal executive offices or mailing address."""
    try:
        import re

        phrases = (
            "principal executive offices are located at",
            "copies of the code are available free of charge by writing to",
            "our headquarters are located in",
        )

        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "text"
            and "item 1" in ((span.get("structure") or {}).get("path_text") or "").lower()
            and any(p in re.sub(r"\s+", " ", (span.get("text") or "").lower()) for p in phrases)
        ]
    except Exception:
        return []
