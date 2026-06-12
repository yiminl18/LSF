def rule_esf_after_foreign_currency_positions(doc: dict) -> list[dict]:
    """Match tables after the Foreign Currency Positions section, where ESF often follows."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") == "section_header" and re.search(r'FOREIGN CURRENCY POSITIONS', span.get("text", ""), re.I):
                for j in range(i + 1, min(len(texts), i + 20)):
                    nxt = texts[j]
                    if nxt.get("label") == "table":
                        out.append(nxt)
                break
        return out
    except Exception:
        return []
