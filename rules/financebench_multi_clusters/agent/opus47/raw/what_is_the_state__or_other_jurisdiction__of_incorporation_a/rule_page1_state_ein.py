def rule_page1_state_ein(doc: dict) -> list[dict]:
    """Match page 1 spans containing state of incorporation and EIN information."""
    try:
        import re
        results = []
        texts = doc.get("texts", [])
        ein_pattern = re.compile(r'\d{2}-\d{7}')

        jurisdictions = {
            "delaware", "washington", "california", "new york", "texas",
            "jersey", "nevada", "florida", "illinois", "massachusetts",
            "maryland", "pennsylvania", "ohio", "georgia", "north carolina",
            "virginia", "colorado", "arizona", "michigan", "minnesota",
        }

        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            text_lower = text.lower().strip()

            if "state or other jurisdiction" in text_lower:
                results.append(span)
                for j in range(1, 4):
                    if i - j >= 0 and texts[i-j].get("page_no") == 1:
                        prev_text = texts[i-j].get("text", "").lower().strip()
                        if prev_text in jurisdictions:
                            results.append(texts[i-j])
                            break
                        for state in jurisdictions:
                            if state in prev_text and len(prev_text) < 50:
                                results.append(texts[i-j])
                                break

            if "employer identification" in text_lower:
                results.append(span)
                if not ein_pattern.search(text) and i > 0 and texts[i-1].get("page_no") == 1:
                    results.append(texts[i-1])

            if ein_pattern.search(text):
                results.append(span)

            if text_lower in jurisdictions:
                results.append(span)

        seen = set()
        unique = []
        for s in results:
            sid = id(s)
            if sid not in seen:
                seen.add(sid)
                unique.append(s)
        return unique
    except Exception:
        return []
