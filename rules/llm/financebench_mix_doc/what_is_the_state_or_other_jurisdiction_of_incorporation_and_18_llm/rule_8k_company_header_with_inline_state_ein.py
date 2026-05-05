def rule_8k_company_header_with_inline_state_ein(doc: dict) -> list[dict]:
    """Match 8-K company H1 blocks that inline state and EIN on the same cover span."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and re.search(r"\b\d{2}-\d{7}\b", text)
                and re.search(r"state or other jurisdiction|incorporation", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
