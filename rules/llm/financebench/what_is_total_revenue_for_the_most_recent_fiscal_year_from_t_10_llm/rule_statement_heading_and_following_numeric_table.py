def rule_statement_heading_and_following_numeric_table(doc: dict) -> list[dict]:
    """Match numeric tables immediately following statement headings for income/operations/earnings."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if not any(k in txt for k in ["income", "operations", "earnings"]):
                continue
            for j in range(i + 1, min(i + 4, len(texts))):
                nxt = texts[j]
                if nxt.get("label") == "table":
                    out.append(nxt)
                    break
        return out
    except Exception:
        return []
