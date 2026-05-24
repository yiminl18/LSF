def rule_page1_combined_state_and_ein_same_text(doc: dict) -> list[dict]:
    """Match page-1 spans whose text itself contains both a jurisdiction value and an EIN number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r"\d{2}-\d{7}", text):
                if re.search(r"[A-Za-z].+\d{2}-\d{7}", text):
                    out.append(span)
        return out
    except Exception:
        return []
