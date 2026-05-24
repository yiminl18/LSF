def rule_page1_cover_text_with_multiple_large_numbers(doc: dict) -> list[dict]:
    """Match cover-page text spans containing multiple large numbers, one of which may be the answer."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") not in (1, 2):
                continue
            text = (span.get("text") or "")
            nums = re.findall(r"\d[\d,]{5,}", text)
            if len(nums) >= 2 and span.get("label") in ("text", "section_header"):
                out.append(span)
        return out
    except Exception:
        return []
