def rule_path_text_equals_text_h1_page1(doc: dict) -> list[dict]:
    """Match page-1 H1 section headers where path_text equals text, a common company-name heading pattern."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            path = (span.get("structure", {}).get("path_text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and txt
                and txt == path
            ):
                out.append(span)
        return out
    except Exception:
        return []
