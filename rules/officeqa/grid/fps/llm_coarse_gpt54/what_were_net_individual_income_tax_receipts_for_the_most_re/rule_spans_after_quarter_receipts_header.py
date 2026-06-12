def rule_spans_after_quarter_receipts_header(doc: dict) -> list[dict]:
    """Match text/table spans immediately after quarter receipts headers, where the answer may be stated or tabulated."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if span.get("label") == "section_header" and re.search(r'Fourth-Quarter Receipts|First-Quarter Receipts', txt, re.I):
                for j in range(i + 1, min(i + 10, len(texts))):
                    s2 = texts[j]
                    if s2.get("label") in {"text", "table"}:
                        out.append(s2)
    except Exception:
        return []
    return out
