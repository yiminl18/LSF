def rule_inline_coverpage_pair_block(doc: dict) -> list[dict]:
    """Match page-1 cover-page spans that inline both state and EIN values in one long block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if (
                span.get("page_no") == 1
                and len(text) > 80
                and re.search(r"state or other jurisdiction of incorporation", text, re.I)
                and re.search(r"\b\d{2}-\d{7}\b", text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
