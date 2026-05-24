def rule_coverpage_sequence_after_exact_name(doc: dict) -> list[dict]:
    """Match a small window of spans following the exact-name label on page 1."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"exact name of registrant|exact name of registrant as specified in (its )?charter", text, re.I):
                for j in range(i + 1, min(i + 8, len(texts))):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
