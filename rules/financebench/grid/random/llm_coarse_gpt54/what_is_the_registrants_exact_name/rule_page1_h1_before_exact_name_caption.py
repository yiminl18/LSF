def rule_page1_h1_before_exact_name_caption(doc: dict) -> list[dict]:
    """Match page-1 H1 section headers whose following nearby spans include the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                window = texts[i + 1:i + 6]
                if any("exact name of registrant as specified in its charter" in ((w.get("text") or "").lower()) for w in window):
                    out.append(span)
        return out
    except Exception:
        return []
