def rule_selected_financial_data_section(doc: dict) -> list[dict]:
    """Match section headers for Selected Financial Data, which often contain a net income row nearby."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("label") == "section_header"
            and re.search(r"selected (financial data|consolidated financial data)", s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
