def rule_page1_before_documents_incorporated(doc: dict) -> list[dict]:
    """Match page-1 outstanding-share spans that occur shortly before documents-incorporated text."""
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            t = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "outstanding" in t and ("common stock" in t or "shares" in t):
                nxt = " ".join((n.get("text") or "") for n in texts[i + 1:i + 8]).lower()
                if "documents incorporated" in nxt:
                    out.append(span)
    except Exception:
        return []
    return out
