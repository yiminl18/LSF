def rule_early_8k_exhibit_tables(doc: dict) -> list[dict]:
    """Match page-1/4 8-K exhibit tables under Item 9.01 or nearby exhibit blocks."""
    try:
        import re

        texts = doc.get("texts", [])
        hits: list[dict] = []
        for idx, span in enumerate(texts):
            if span.get("label") != "table" or (span.get("page_no") or 99) > 4:
                continue

            text = " ".join((span.get("text") or "").split())
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            has_exhibit_header = (
                re.search(r"\bexhibit\s*(?:number|no\.?)\b", text, re.IGNORECASE)
                or re.search(r"\|\s*exhibit\s*\|\s*description\s*\|", text, re.IGNORECASE)
            )
            if not has_exhibit_header:
                continue

            nearby = " ".join(
                " ".join((texts[j].get("text") or "").split())
                for j in range(max(0, idx - 5), idx + 1)
                if (texts[j].get("page_no") or 99) <= 4
            ).lower()

            if "item 9.01" in path or "item 8.01" in path or "item 9.01" in nearby or "(d) exhibits" in nearby:
                hits.append(span)
        return hits
    except Exception:
        return []
