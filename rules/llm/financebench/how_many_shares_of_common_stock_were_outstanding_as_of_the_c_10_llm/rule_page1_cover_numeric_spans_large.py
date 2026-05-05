def rule_page1_cover_numeric_spans_large(doc: dict) -> list[dict]:
    """Match large comma-formatted numeric spans on page 1/2 under the company cover block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if span.get("page_no") in (1, 2) and re.fullmatch(r"[\d,]{6,}", text):
                if path and "part i" not in path and "item " not in path:
                    out.append(span)
        return out
    except Exception:
        return []
