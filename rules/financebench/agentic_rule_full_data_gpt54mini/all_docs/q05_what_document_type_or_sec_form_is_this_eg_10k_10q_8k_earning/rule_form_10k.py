def rule_form_10k(doc: dict) -> list[dict]:
    try:
        import re

        text = doc.get("text", "") or ""
        prefix = text[:6000]
        patterns = [
            r"FORM\s*10\s*[-]?\s*K(?:/A)?",
            r"ANNUAL REPORT PURSUANT TO SECTION 13 OR 15\(D\)",
        ]
        for pattern in patterns:
            m = re.search(pattern, prefix, flags=re.IGNORECASE)
            if not m:
                continue
            start = max(0, m.start() - 80)
            end = min(len(prefix), m.end() + 120)
            return [{"text": prefix[start:end]}]
        return []
    except Exception:
        return []
