def rule_exchange_after_name_of_exchange_header(doc: dict) -> list[dict]:
    """Match the next few spans after a standalone 'Name of each exchange on which registered' header."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            if re.fullmatch(r'Name of each exchange on which registered', (s.get("text") or "").strip(), re.I):
                for j in range(i + 1, min(len(texts), i + 5)):
                    t = (texts[j].get("text") or "").strip()
                    if re.search(r'new york stock exchange|nasdaq|global select market', t, re.I):
                        out.append(texts[j])
        return out
    except Exception:
        return []
