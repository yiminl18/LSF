def rule_notice_item_number_text_lines(doc: dict) -> list[dict]:
    """Match short notice text lines that read Item number or Item number N."""
    try:
        import re

        line_re = re.compile(r"^\s*Item number(?:\s+\d+)?\s*$", re.IGNORECASE)
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
            if span.get("label") in ("text", "list_item")
            and line_re.match(" ".join((span.get("text") or "").split()))
        ]
    except Exception:
        return []
