def rule_table_with_in_millions_and_receipts(doc: dict) -> list[dict]:
    """Match tables near '(In millions of dollars)' and receipts headers."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            if not re.search(r'(Net receipts|Total receipts|Net budget receipts)', txt, re.I):
                continue
            window = texts[max(0, i-3):i]
            nearby = " ".join((s.get("text") or "") for s in window)
            if re.search(r'In millions? of dollars|In millions', nearby, re.I):
                out.append(span)
        return out
    except Exception:
        return []
