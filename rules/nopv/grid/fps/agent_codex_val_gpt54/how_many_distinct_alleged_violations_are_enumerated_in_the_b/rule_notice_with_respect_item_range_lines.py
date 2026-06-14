def rule_notice_with_respect_item_range_lines(doc: dict) -> list[dict]:
    """Match notice summary lines that enumerate an inclusive item range."""
    try:
        import re

        line_re = re.compile(
            r"^\s*With respect to items?\s+\d+\s+(?:to|through)\s+\d+\b",
            re.IGNORECASE,
        )
        stop_re = re.compile(r"^\s*Response to this Notice\s*$", re.IGNORECASE)

        texts = doc.get("texts", [])
        cutoff = len(texts)
        for i, span in enumerate(texts):
            text = " ".join((span.get("text") or "").split())
            if stop_re.match(text):
                cutoff = i
                break

        return [
            span
            for span in texts[:cutoff]
            if span.get("label") == "text"
            and line_re.match(" ".join((span.get("text") or "").split()))
        ]
    except Exception:
        return []
