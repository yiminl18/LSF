def rule_page1_cover_identity_cluster(doc: dict) -> list[dict]:
    """Match spans in the dense page-1 identity cluster containing company/address/EIN/phone details."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            window = texts[max(0, i-3): min(len(texts), i+4)]
            context = " ".join(((w.get("text") or "") + " " + (w.get("text_span") or "")) for w in window).lower()
            if "exact name of registrant" in context and "employer identification" in context:
                out.append(span)
        return out
    except Exception:
        return []
