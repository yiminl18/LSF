def rule_page123_following_span_after_non191_reporting_fallback(doc: dict) -> list[dict]:
    """When an early Part 191 item appears first, return the first nearby span after the first later non-191 cited section."""
    try:
        import re

        texts = doc.get("texts", [])
        cite_re = re.compile(r"^\s*(?:\d+\s*[\.\)]\s*)?(?:[§s]\s*)?(19\d)\.\d+\b", re.IGNORECASE)
        next_item_re = re.compile(r"^\s*\d+\s*[\.\)]\s*(?:[§s]\s*)?19\d\.\d+\b", re.IGNORECASE)

        first_part = None
        for span in texts:
            if span.get("page_no", 99) > 2:
                continue
            match = cite_re.search((span.get("text") or "").strip())
            if match:
                first_part = match.group(1)
                break
        if first_part != "191":
            return []

        anchor_idx = None
        anchor_page = None
        for i, span in enumerate(texts):
            if span.get("page_no", 99) > 3:
                continue
            match = cite_re.search((span.get("text") or "").strip())
            if match and match.group(1) != "191":
                anchor_idx = i
                anchor_page = span.get("page_no", 99)
                break
        if anchor_idx is None:
            return []

        for j in range(anchor_idx + 1, min(anchor_idx + 6, len(texts))):
            span = texts[j]
            if span.get("page_no", 99) > min(3, anchor_page + 1):
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
