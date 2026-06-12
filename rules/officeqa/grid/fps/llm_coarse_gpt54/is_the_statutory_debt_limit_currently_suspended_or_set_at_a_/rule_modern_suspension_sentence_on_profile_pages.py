def rule_modern_suspension_sentence_on_profile_pages(doc: dict) -> list[dict]:
    """Match modern profile pages where a sentence states the debt ceiling was suspended until a date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"On .* debt ceiling was suspended until .*", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
