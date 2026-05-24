def rule_coverpage_sequence_before_irs_label(doc: dict) -> list[dict]:
    """Match a small window of spans immediately before the IRS label, often including the state value."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = span.get("text", "") or ""
            if re.search(r"i\.?r\.?s\.? employer identification|employer identification no", text, re.I):
                for j in range(max(0, i - 3), i + 1):
                    out.append(texts[j])
        return out
    except Exception:
        return []
