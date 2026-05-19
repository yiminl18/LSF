def rule_inline_registered_under_symbol(doc: dict) -> list[dict]:
    """Match body text that states stock is listed/traded under a symbol."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"(listed|traded|trades)\s+on\s+the\s+.*?(nasdaq|new york stock exchange|nyse).*?under\s+the\s+symbol", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
