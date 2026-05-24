def rule_page1_jurisdiction_ein(doc: dict) -> list[dict]:
    """Match page 1 spans containing state/jurisdiction and EIN info."""
    try:
        import re
        results = []
        texts = doc.get("texts", [])

        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue

            text = span.get("text", "")
            text_span = span.get("text_span", "")
            combined = (text + " " + text_span).lower()

            if any(kw in combined for kw in [
                "jurisdiction", "state of incorporation",
                "employer identification", "i.r.s. employer", "irs employer"
            ]):
                results.append(span)
                if i > 0:
                    prev = texts[i - 1]
                    if prev.get("page_no") == 1:
                        prev_text = prev.get("text", "")
                        if "registrant" not in prev_text.lower() and "exact name" not in prev_text.lower():
                            results.append(prev)

            if re.search(r'\b\d{2}-\d{7}\b', text):
                results.append(span)

        seen = set()
        unique = []
        for s in results:
            key = id(s)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique
    except Exception:
        return []
