def rule_exact_exchange_label_values(doc: dict) -> list[dict]:
    """Match spans that are likely the value under 'Name of each exchange on which registered'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if re.fullmatch(r"(The\s+)?Nasdaq Global Select Market", txt, re.I):
                out.append(span)
            elif re.fullmatch(r"(The\s+)?New York Stock Exchange(?:\s*\(NYSE\))?", txt, re.I):
                out.append(span)
            elif re.fullmatch(r"New York Stock Exchange", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
