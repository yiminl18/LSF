def rule_narrative_filed_as_exhibit_reference(doc: dict) -> list[dict]:
    """Match narrative spans that explicitly say an agreement was filed as a qualifying exhibit."""
    try:
        import re

        filed_as_re = re.compile(
            r"\b(?:filed|furnished)\s+as\s+exhibit\s+(10\s*\.\s*\d+|4\s*\.\s*\d+|99\s*\.\s*\d+|104)\b",
            re.I,
        )
        return [
            span for span in doc.get("texts", [])
            if span.get("label") in {"text", "list_item"}
            and filed_as_re.search(" ".join((span.get("text") or "").split()))
        ]
    except Exception:
        return []
