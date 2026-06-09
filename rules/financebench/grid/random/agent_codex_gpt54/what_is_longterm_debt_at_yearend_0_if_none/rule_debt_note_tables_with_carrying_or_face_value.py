def rule_debt_note_tables_with_carrying_or_face_value(doc: dict) -> list[dict]:
    """Match debt-note tables that state carrying value, face value, or net carrying amount of long-term debt."""
    try:
        hits: list[dict] = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = " ".join((span.get("text") or "").split()).lower()
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            if "debt" not in path:
                continue
            if (
                "carrying value of long-term debt" in text
                or "face value of long-term debt" in text
                or "total net carrying amount" in text
                or "long-term debt, excluding current portion" in text
                or "debt, non-current" in text
            ):
                hits.append(span)
        return hits
    except Exception:
        return []
