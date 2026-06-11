def rule_page1_text_with_state_label_only(doc: dict) -> list[dict]:
    """Match page-1 spans containing only the state/jurisdiction label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"State or other jurisdiction of incorporation(?: or organization)?|State or other jurisdiction of incorporation\)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
