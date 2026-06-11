def rule_page1_state_name_candidates(doc: dict) -> list[dict]:
    """Match page-1 short bold spans likely to be the incorporation state/jurisdiction."""
    try:
        states = {
            "delaware", "new york", "new jersey", "new mexico", "new hampshire",
            "new mexico", "washington", "california", "minnesota", "jersey"
        }
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip().lower()
            if txt in states and span.get("bold") == 1:
                out.append(span)
        return out
    except Exception:
        return []
