def rule_profile_of_economy_all_text_on_answer_pages(doc: dict) -> list[dict]:
    """Match all text spans on pages that contain Profile of the Economy and consumer-related mentions."""
    import re
    try:
        texts = doc.get("texts", [])
        pages = set()
        for span in texts:
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "profile of the economy" in path.lower() and re.search(r"(consumer|sentiment|confidence|michigan|reuters)", text, re.I):
                if span.get("page_no") is not None:
                    pages.add(span.get("page_no"))
        return [span for span in texts if span.get("page_no") in pages and span.get("label") in {"text", "table", "section_header"}]
    except Exception:
        return []
