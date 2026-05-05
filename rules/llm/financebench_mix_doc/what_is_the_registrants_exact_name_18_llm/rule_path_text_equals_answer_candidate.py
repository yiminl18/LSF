def rule_path_text_equals_answer_candidate(doc: dict) -> list[dict]:
    """Match page-1 H1 spans where path_text equals the span text, a common company-name header pattern."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            path = (span.get("structure", {}).get("path_text", "") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and txt
                and txt == path
                and "form 10-" not in txt.lower()
                and "form 8-k" not in txt.lower()
                and "current report" not in txt.lower()
                and "securities and exchange commission" not in txt.lower()
            ):
                out.append(span)
        return out
    except Exception:
        return []
