def rule_item2_properties_headquarters_sentence(doc: dict) -> list[dict]:
    """Match Item 2 Properties sentences that explicitly state the headquarters or principal office location."""
    try:
        import re

        phrases = (
            "principal corporate and administrative offices",
            "corporate headquarters is located in",
            "headquarters are located in",
            "principal executive offices are located at",
        )

        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "text"
            and "item 2" in ((span.get("structure") or {}).get("path_text") or "").lower()
            and "properties" in ((span.get("structure") or {}).get("path_text") or "").lower()
            and any(p in re.sub(r"\s+", " ", (span.get("text") or "").lower()) for p in phrases)
        ]
    except Exception:
        return []
