def rule_page13_bold_cover_title_form_code(doc: dict) -> list[dict]:
    """Match short bold cover titles on page 1-3 that contain a FORM 10-K, 10-Q, or 8-K code."""
    try:
        out = []
        for s in doc.get("texts", []):
            text = " ".join((s.get("text") or "").split())
            if (
                (s.get("page_no") or 99) <= 3
                and s.get("label") in {"text", "section_header"}
                and s.get("bold") == 1
                and len(text) <= 120
                and (
                    "FORM 10-K" in text.upper()
                    or "FORM 10-Q" in text.upper()
                    or "FORM 8-K" in text.upper()
                )
            ):
                out.append(s)
        return out
    except Exception:
        return []
