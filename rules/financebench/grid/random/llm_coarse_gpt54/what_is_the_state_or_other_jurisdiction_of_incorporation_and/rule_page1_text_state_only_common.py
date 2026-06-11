def rule_page1_text_state_only_common(doc: dict) -> list[dict]:
    """Match page-1 short bold/text spans that are likely the state/jurisdiction value."""
    try:
        import re
        states = {
            "delaware", "washington", "new york", "jersey", "jersey (channel islands)",
            "california", "nevada", "virginia", "massachusetts", "ohio", "texas"
        }
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").strip()
            low = text.lower()
            if span.get("page_no") == 1 and low in states:
                out.append(span)
            elif span.get("page_no") == 1 and re.fullmatch(r"[A-Z][A-Za-z]+(?:\s*\([A-Za-z ]+\))?", text) and span.get("bold") == 1:
                out.append(span)
        return out
    except Exception:
        return []
