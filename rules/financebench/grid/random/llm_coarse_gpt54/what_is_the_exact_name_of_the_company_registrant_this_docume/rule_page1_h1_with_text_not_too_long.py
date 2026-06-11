def rule_page1_h1_with_text_not_too_long(doc: dict) -> list[dict]:
    """Match concise page-1 H1 spans likely to be company names rather than long narrative headers."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and 2 <= len(txt.split()) <= 8
                and len(txt) <= 80
                and "form 10-" not in txt.lower()
                and "form 8-k" not in txt.lower()
                and "securities and exchange commission" not in txt.lower()
            ):
                out.append(span)
        return out
    except Exception:
        return []
