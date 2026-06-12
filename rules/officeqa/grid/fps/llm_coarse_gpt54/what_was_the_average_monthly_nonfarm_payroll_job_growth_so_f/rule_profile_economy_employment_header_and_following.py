def rule_profile_economy_employment_header_and_following(doc: dict) -> list[dict]:
    """Return the employment/labor header plus following text spans likely containing the answer."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") == "section_header" and re.search(r'(Employment and unemployment|Labor Markets and Wages|Labor Markets)', span.get("text") or "", re.I):
                out.append(span)
                for j in range(i + 1, min(i + 4, len(texts))):
                    s = texts[j]
                    if s.get("label") == "text":
                        out.append(s)
        return out
    except Exception:
        return []
