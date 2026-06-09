def rule_tables_after_exhibit_index_headers(doc: dict) -> list[dict]:
    """Match tables that follow Exhibit Index, Index to Exhibits, or numbered Exhibits headers."""
    try:
        import re

        texts = doc.get("texts", [])
        hits: list[dict] = []
        seen: set[int] = set()

        def is_anchor(span: dict) -> bool:
            text = " ".join((span.get("text") or "").split()).lower()
            norm = re.sub(r"[^a-z0-9.]+", " ", text).strip()
            return (
                "exhibit index" in norm
                or "index to exhibits" in norm
                or norm in {"3. exhibits", "3 exhibits", "(3) exhibits", "2. exhibits", "2 exhibits", "(2) exhibits"}
            )

        def is_exhibit_table(span: dict) -> bool:
            text = " ".join((span.get("text") or "").split())
            return span.get("label") == "table" and (
                re.search(r"\bexhibit\s*(?:number|no\.?)\b", text, re.IGNORECASE)
                or re.search(r"\|\s*exhibit\s*\|\s*description\s*\|", text, re.IGNORECASE)
                or re.search(r"\|\s*exhibit number\s*\|\s*exhibit description\s*\|", text, re.IGNORECASE)
                or re.search(r"\|\s*no\.\s*\|\s*exhibit description\s*\|", text, re.IGNORECASE)
                or re.search(r"\|\s*\*{0,2}(?:10|4|99|104|3)\s*\.\s*\d", text)
            )

        for idx, span in enumerate(texts):
            if not is_anchor(span):
                continue
            anchor_page = span.get("page_no") or 0
            for j in range(idx + 1, min(len(texts), idx + 16)):
                candidate = texts[j]
                page_no = candidate.get("page_no") or 0
                if page_no and anchor_page and page_no - anchor_page > 4:
                    break
                if is_exhibit_table(candidate) and j not in seen:
                    hits.append(candidate)
                    seen.add(j)
        return hits
    except Exception:
        return []
