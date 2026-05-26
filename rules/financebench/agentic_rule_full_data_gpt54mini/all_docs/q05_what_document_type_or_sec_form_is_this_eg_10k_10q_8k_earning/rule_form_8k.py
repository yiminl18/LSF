def rule_form_8k(doc: dict) -> list[dict]:
    try:
        import re

        text = doc.get("text", "") or ""
        prefix = text[:6000]
        patterns = [
            r"FORM\s*8\s*[-]?\s*K(?:/A)?",
            r"CURRENT REPORT PURSUANT TO SECTION 13 OR 15\(D\)",
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
