def rule_early_narrative_principal_offices_sentence(doc: dict) -> list[dict]:
    """Match early narrative sentences that explicitly state the headquarters or principal offices location."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 15:
                continue
            if span.get("label") != "text":
                continue

            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            if (
                "principal executive offices are located at" in lowered
                or "our principal executive offices are located at" in lowered
                or "headquartered in" in lowered
            ):
                results.append(span)

        return results
    except Exception:
        return []
