def rule_note_debt_header_near_table(doc: dict) -> list[dict]:
    """Match section headers for debt notes that likely precede the answer table."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt) or "notes payable" in txt:
                for j in range(i + 1, min(i + 6, len(texts))):
                    nxt = texts[j]
                    if nxt.get("label") == "table":
                        out.append(nxt)
                        break
        return out
    except Exception:
        return []
