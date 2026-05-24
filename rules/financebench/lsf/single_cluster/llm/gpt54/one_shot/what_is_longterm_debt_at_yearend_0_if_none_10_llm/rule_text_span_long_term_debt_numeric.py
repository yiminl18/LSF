def rule_text_span_long_term_debt_numeric(doc: dict) -> list[dict]:
    """Match text spans containing long-term debt with nearby numeric amount."""
    import re
    try:
        out = []
        num_re = re.compile(r"[$]?\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion))?")
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "section_header", "list_item"}:
                continue
            txt = span.get("text") or ""
            low = txt.lower()
            if re.search(r"\blong[\-\s]?term debt\b", low) and num_re.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
