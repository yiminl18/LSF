def rule_canadian_header_and_following_table_same_page(doc: dict) -> list[dict]:
    """Match tables on the same page as a Canadian dollar positions header."""
    try:
        texts = doc.get("texts", [])
        pages = set()
        for span in texts:
            if "canadian dollar positions" in (span.get("text") or "").lower():
                p = span.get("page_no")
                if isinstance(p, int):
                    pages.add(p)
        out = []
        for span in texts:
            if span.get("label") == "table" and span.get("page_no") in pages:
                out.append(span)
        return out
    except Exception:
        return []
