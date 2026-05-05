def rule_page1_bold_state_name_candidates(doc: dict) -> list[dict]:
    """Match bold page-1 short spans that are likely state/jurisdiction values near the cover block."""
    try:
        import re
        stop = {
            "form 10-k", "part i", "none", "yes", "no", "common stock", "table of contents",
            "new york stock exchange", "nasdaq global select market", "glw", "nke", "lmt"
        }
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("bold") == 1 and 1 <= len(text.split()) <= 4:
                low = text.lower()
                if low not in stop and not re.search(r"\d", text):
                    if span.get("label") in ("text", "section_header"):
                        out.append(span)
        return out
    except Exception:
        return []
