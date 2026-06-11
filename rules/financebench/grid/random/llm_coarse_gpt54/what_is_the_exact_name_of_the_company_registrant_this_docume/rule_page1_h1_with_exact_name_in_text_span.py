def rule_page1_h1_with_exact_name_in_text_span(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose text_span itself contains exact name of registrant language."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and "exact name of registrant" in (span.get("text_span") or "").lower()
            ):
                out.append(span)
        return out
    except Exception:
        return []
