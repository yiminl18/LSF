def rule_coverpage_sequence_before_state_label(doc: dict) -> list[dict]:
    """Match a small window of spans immediately before the state/jurisdiction label, often including the state value."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = span.get("text", "") or ""
            if re.search(r"state or other jurisdiction of incorporation|state or other jurisdiction of incorporation or organization", text, re.I):
                for j in range(max(0, i - 2), i + 1):
                    out.append(texts[j])
        return out
    except Exception:
        return []
