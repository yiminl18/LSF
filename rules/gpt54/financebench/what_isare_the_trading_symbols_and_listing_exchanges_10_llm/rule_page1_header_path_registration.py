def rule_page1_header_path_registration(doc: dict) -> list[dict]:
    """Match page 1 spans whose path_text itself contains registration-related terms."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("page_no") == 1 and re.search(r"trading symbol|exchange|12\(b\)|common stock|registered", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
