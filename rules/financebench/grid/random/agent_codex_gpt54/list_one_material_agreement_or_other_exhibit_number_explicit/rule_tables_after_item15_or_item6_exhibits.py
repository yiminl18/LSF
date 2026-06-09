def rule_tables_after_item15_or_item6_exhibits(doc: dict) -> list[dict]:
    """Match exhibit tables that appear immediately after Item 15 or Item 6 exhibit-section anchors."""
    try:
        import re

        texts = doc.get("texts", [])
        hits: list[dict] = []
        seen: set[int] = set()

        def is_anchor(span: dict) -> bool:
            text = " ".join((span.get("text") or "").split()).lower()
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            blob = f"{path} {text}"
            return (
                ("item 15" in blob and "exhibit" in blob)
                or ("item 6" in blob and "exhibit" in blob)
            )

        def keep_table(span: dict) -> bool:
            text = " ".join((span.get("text") or "").split())
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            return span.get("label") == "table" and (
                "exhibit" in path
                or re.search(r"\bexhibit\s*(?:number|no\.?)\b", text, re.IGNORECASE)
                or re.search(r"\|\s*exhibit\s*\|\s*description\s*\|", text, re.IGNORECASE)
                or re.search(r"\|\s*no\.\s*\|\s*exhibit description\s*\|", text, re.IGNORECASE)
                or re.search(r"\|\s*10\s*\.\s*\d", text)
            )

        for idx, span in enumerate(texts):
            if not is_anchor(span):
                continue
            anchor_page = span.get("page_no") or 0
            for j in range(idx + 1, min(len(texts), idx + 18)):
                candidate = texts[j]
                page_no = candidate.get("page_no") or 0
                if page_no and anchor_page and page_no - anchor_page > 4:
                    break
                if keep_table(candidate) and j not in seen:
                    hits.append(candidate)
                    seen.add(j)
        return hits
    except Exception:
        return []
