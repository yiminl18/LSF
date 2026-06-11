def rule_exchange_in_text_when_path_mentions_ticker(doc: dict) -> list[dict]:
    """Match exchange spans whose path_text includes a ticker-like short header context."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if re.search(r'\b[A-Z]{2,5}\b', path) and re.search(r'new york stock exchange|nasdaq|global select market', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
