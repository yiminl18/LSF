def rule_apple_note_list_exchange_cluster(doc: dict) -> list[dict]:
    """Match Apple-style page-1 clusters where many note titles are followed by exchange names."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            txt = (s.get("text") or "")
            if s.get("page_no") == 1 and re.search(r"Notes due \d{4}|Common Stock", txt, re.I):
                window = texts[i:min(len(texts), i + 12)]
                joined = " ".join((w.get("text") or "") for w in window)
                if re.search(r"Nasdaq|New York Stock Exchange", joined, re.I):
                    out.extend(window)
        return out
    except Exception:
        return []
