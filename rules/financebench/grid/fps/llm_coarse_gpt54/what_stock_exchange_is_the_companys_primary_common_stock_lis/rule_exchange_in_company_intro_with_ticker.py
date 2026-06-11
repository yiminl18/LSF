def rule_exchange_in_company_intro_with_ticker(doc: dict) -> list[dict]:
    """Match company intro sentences that include ticker and exchange together."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'\((nasdaq|nyse)[: ]', combined, re.I) or re.search(r'under the symbol', combined, re.I):
                if re.search(r'new york stock exchange|nasdaq|global select market', combined, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
