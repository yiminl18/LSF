def rule_page1_identifier_five_span_window(doc: dict) -> list[dict]:
    """Match broader five-span windows on page 1 containing the state/EIN answer cluster."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 4):
            window = texts[i:i+5]
            if all(s.get("page_no") == 1 for s in window):
                joined = " ".join((s.get("text", "") or "") for s in window)
                if re.search(r"\b\d{2}-\d{7}\b", joined) and re.search(r"state or other jurisdiction|incorporation|employer identification|\b(Delaware|Washington|New York|Jersey)\b", joined, re.I):
                    out.extend(window)
        return out
    except Exception:
        return []
