def rule_currently_listed_on_exchange_sentence(doc: dict) -> list[dict]:
    """Match narrative spans saying stock is 'currently listed on' an exchange."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'currently listed on .*?(new york stock exchange|nasdaq)', combined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
