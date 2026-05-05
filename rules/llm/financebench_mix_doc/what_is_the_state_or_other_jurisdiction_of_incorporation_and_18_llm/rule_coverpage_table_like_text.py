def rule_coverpage_table_like_text(doc: dict) -> list[dict]:
    """Match page-1 table or table-like spans near the company header that may contain the answer."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") in {"table", "section_header", "text"}:
                text = span.get("text", "") or ""
                if "|" in text or "State or other jurisdiction" in text or "Employer Identification" in text:
                    out.append(span)
        return out
    except Exception:
        return []
