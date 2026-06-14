def rule_notice_with_respect_single_item_lines(doc: dict) -> list[dict]:
    """Match notice summary lines that reference exactly one numbered item."""
    try:
        import re

        line_re = re.compile(
            r"^\s*With respect to item\s*,?\s*\d+\b(?!\s*(?:,?\s*\d|\s*(?:and|to|through)\b))",
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

        out = []
        for span in texts[:cutoff]:
            if span.get("label") != "text":
                continue
            text = " ".join((span.get("text") or "").split())
            if line_re.match(text):
                out.append(span)
        return out
    except Exception:
        return []
