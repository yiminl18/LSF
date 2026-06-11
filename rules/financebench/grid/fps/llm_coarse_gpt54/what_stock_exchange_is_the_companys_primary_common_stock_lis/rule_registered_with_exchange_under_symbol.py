def rule_registered_with_exchange_under_symbol(doc: dict) -> list[dict]:
    """Match narrative spans saying shares are traded/registered on an exchange under a symbol."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'(traded|registered|listed) on the .*?(new york stock exchange|nasdaq).*?under the symbol', combined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
