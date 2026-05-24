def rule_page1_exchange_in_registration_window(doc: dict) -> list[dict]:
    """Match exchange-name spans appearing after the first 12(b) mention and before 12(g) on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        start = None
        end = None
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if start is None and span.get("page_no") == 1 and re.search(r"12\(b\)", txt):
                start = i
            if start is not None and span.get("page_no") == 1 and re.search(r"12\(g\)", txt):
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 20)
        out = []
        for s in texts[start:end]:
            t = (s.get("text") or "")
            if re.search(r"nasdaq|new york stock exchange|nyse|australian securities exchange", t, re.I):
                out.append(s)
        return out
    except Exception:
        return []
