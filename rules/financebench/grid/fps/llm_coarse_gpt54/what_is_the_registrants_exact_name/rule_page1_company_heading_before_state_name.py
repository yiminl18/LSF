def rule_page1_company_heading_before_state_name(doc: dict) -> list[dict]:
    """Match the nearest prominent span before a state-name span in the registrant block."""
    try:
        texts = doc.get("texts", [])
        states = {"delaware", "new york", "washington", "minnesota", "new jersey", "california", "jersey"}
        out = []
        for i, span in enumerate(texts):
            low = (span.get("text") or "").strip().lower()
            if span.get("page_no") == 1 and low in states:
                for j in range(i - 1, max(-1, i - 5), -1):
                    prev = texts[j]
                    if prev.get("page_no") == 1 and prev.get("bold") == 1 and float(prev.get("size") or 0) >= 10:
                        out.append(prev)
                        return out
        return out
    except Exception:
        return []
