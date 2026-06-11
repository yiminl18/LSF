def rule_page1_company_name_near_state_value(doc: dict) -> list[dict]:
    """Match page-1 span near a standalone state value like Delaware/New York/California/Minnesota/etc."""
    try:
        texts = doc.get("texts", [])
        states = {
            "delaware", "new york", "california", "minnesota", "washington",
            "new jersey", "jersey", "virginia"
        }
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            window = texts[i:i+8]
            vals = " ".join((w.get("text", "") or "").lower() for w in window)
            if any(st in vals for st in states):
                if "form 10-" not in txt.lower() and "form 8-k" not in txt.lower() and "current report" not in txt.lower():
                    out.append(span)
        return out
    except Exception:
        return []
