def rule_page1_numeric_only_after_outstanding_label(doc: dict) -> list[dict]:
    """Match numeric-only spans on page 1/2 that follow a label span mentioning outstanding common stock."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            t = (span.get("text") or "")
            if span.get("page_no") not in (1, 2):
                continue
            if re.fullmatch(r"[\$]?\s*[\d,]{6,}", t.strip()):
                window = texts[max(0, i - 3):i]
                ctx = " ".join((w.get("text") or "").lower() for w in window if w.get("page_no") in (1, 2))
                if "outstanding" in ctx and ("common stock" in ctx or "shares" in ctx):
                    out.append(span)
        return out
    except Exception:
        return []
