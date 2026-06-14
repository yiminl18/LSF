def rule_page12_first_following_span_after_part_anchor(doc: dict) -> list[dict]:
    """Return the first nearby non-empty span after the earliest cited 49 CFR section on pages 1-2."""
    try:
        import re

        texts = doc.get("texts", [])
        cite_re = re.compile(r"^\s*(?:\d+\s*[\.\)]\s*)?(?:[§s]\s*)?19\d\.\d+\b", re.IGNORECASE)
        next_item_re = re.compile(r"^\s*\d+\s*[\.\)]\s*(?:[§s]\s*)?19\d\.\d+\b", re.IGNORECASE)

        anchor_idx = None
        anchor_page = None
        for i, span in enumerate(texts):
            if span.get("page_no", 99) > 2:
                continue
            if cite_re.search((span.get("text") or "").strip()):
                anchor_idx = i
                anchor_page = span.get("page_no", 99)
                break
        if anchor_idx is None:
            return []

        for j in range(anchor_idx + 1, min(anchor_idx + 6, len(texts))):
            span = texts[j]
            if span.get("page_no", 99) > min(2, anchor_page + 1):
                break
            text = (span.get("text") or "").strip()
            if not text:
                continue
            if next_item_re.search(text):
                break
            return [span]
        return []
    except Exception:
        return []
