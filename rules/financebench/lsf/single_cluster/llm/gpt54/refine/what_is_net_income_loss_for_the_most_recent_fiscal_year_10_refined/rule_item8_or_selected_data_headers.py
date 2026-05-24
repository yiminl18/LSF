def rule_item8_or_selected_data_headers(doc: dict) -> list[dict]:
    """Match headers for Item 8 or Selected Financial Data as anchors for nearby answer spans."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("label") == "section_header"
            and re.search(r"(item\s*8|selected (financial|consolidated financial) data|statement of income|statement of operations)", s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
