def rule_notice_with_respect_item_list_lines(doc: dict) -> list[dict]:
    """Match notice summary lines that enumerate multiple item numbers as a list."""
    try:
        import re

        start_re = re.compile(r"^\s*With respect to items?\s+\d+", re.IGNORECASE)
        range_re = re.compile(r"\bitems?\s+\d+\s+(?:to|through)\s+\d+\b", re.IGNORECASE)
        list_re = re.compile(r"\b(?:and|,)\s*\d+\b", re.IGNORECASE)
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
            if not start_re.match(text):
                continue
            if range_re.search(text):
                continue
            if list_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
