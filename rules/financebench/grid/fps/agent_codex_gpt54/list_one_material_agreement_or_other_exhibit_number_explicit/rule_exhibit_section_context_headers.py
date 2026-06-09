def rule_exhibit_section_context_headers(doc: dict) -> list[dict]:
    """Match short exhibit-section headers and intro lines that frame the Exhibit Index."""
    try:
        import re

        hits: list[dict] = []
        for span in doc.get("texts", []):
            text = " ".join((span.get("text") or "").split())
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split())
            blob = f"{path} {text}".lower()
            if len(text) > 240:
                continue
            if span.get("label") not in {"text", "section_header", "list_item"}:
                continue
            if re.search(r"\bsignatures?\b|\bcertification\b", text, re.IGNORECASE):
                continue
            text_l = text.lower()
            if (
                ("item 15" in text_l and "exhibit" in text_l)
                or ("item 16" in text_l and "exhibit" in text_l)
                or ("item 6" in text_l and "exhibit" in text_l)
                or "exhibit index" in text_l
                or "index to exhibits" in text_l
                or re.fullmatch(r"\(?3\)?\.?\s+exhibits", text_l)
                or re.fullmatch(r"\(?2\)?\.?\s+exhibits", text_l)
                or "the exhibits listed" in text_l
                or "documents in the accompanying exhibits index" in text_l
                or ("item 9.01" in blob and re.search(r"\bexhibits?\b", text, re.IGNORECASE))
                or re.fullmatch(r"\(?d\)?\.?\s+exhibits\.?", text_l)
            ):
                hits.append(span)
        return hits
    except Exception:
        return []
