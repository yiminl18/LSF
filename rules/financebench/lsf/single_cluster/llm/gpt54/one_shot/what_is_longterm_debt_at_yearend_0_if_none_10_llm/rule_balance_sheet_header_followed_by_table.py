def rule_balance_sheet_header_followed_by_table(doc: dict) -> list[dict]:
    """Match the first table shortly after a Consolidated Balance Sheet header."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if "balance sheet" in txt:
                for j in range(i + 1, min(i + 8, len(texts))):
                    nxt = texts[j]
                    if nxt.get("label") == "table":
                        out.append(nxt)
                        break
        return out
    except Exception:
        return []
