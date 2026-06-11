def rule_page1_before_state_or_incorporation(doc: dict) -> list[dict]:
    """Match the prominent span immediately before a state/incorporation label on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            low = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "state or other jurisdiction of incorporation" in low or low.strip() in {
                "delaware", "new york", "washington", "minnesota", "new jersey", "california", "jersey"
            }:
                for j in range(max(0, i - 3), i):
                    prev = texts[j]
                    ptxt = (prev.get("text") or "").strip()
                    if prev.get("page_no") == 1 and prev.get("bold") == 1 and float(prev.get("size") or 0) >= 10:
                        if "exact name of registrant" not in ptxt.lower():
                            out.append(prev)
                break
        return out
    except Exception:
        return []
