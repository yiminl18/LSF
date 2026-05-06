def rule_page1_h1_or_h2_with_nearby_exact_name(doc: dict) -> list[dict]:
    """Match page-1 H1/H2 headings that have the exact-name caption within the next few spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if span.get("structure", {}).get("level") not in {"H1", "H2"}:
                continue
            txt = (span.get("text") or "").strip().lower()
            if "form 10-k" in txt or "commission" in txt or txt == "or":
                continue
            window = texts[i + 1:i + 6]
            if any("exact name of registrant" in ((w.get("text") or "").lower()) for w in window):
                out.append(span)
        return out
    except Exception:
        return []
