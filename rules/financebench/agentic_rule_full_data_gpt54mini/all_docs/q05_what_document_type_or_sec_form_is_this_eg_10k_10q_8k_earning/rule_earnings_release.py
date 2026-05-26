def rule_earnings_release(doc: dict) -> list[dict]:
    try:
        import re

        text = doc.get("text", "") or ""
        prefix = text[:6000]
        patterns = [
            r"EARNINGS RELEASE",
            r"PRESS RELEASE",
            r"NEWS RELEASE",
            r"EARNINGS CALL",
            r"RESULTS OF OPERATIONS AND FINANCIAL CONDITION",
        ]
        for pattern in patterns:
            m = re.search(pattern, prefix, flags=re.IGNORECASE)
            if not m:
                continue
            start = max(0, m.start() - 80)
            end = min(len(prefix), m.end() + 140)
            return [{"text": prefix[start:end]}]
        return []
    except Exception:
        return []
