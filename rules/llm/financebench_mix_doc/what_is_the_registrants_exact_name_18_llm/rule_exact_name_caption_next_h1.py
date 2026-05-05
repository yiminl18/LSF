def rule_exact_name_caption_next_h1(doc: dict) -> list[dict]:
    """Match H1/section_header spans immediately followed by the '(Exact name of registrant...)' caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "exact name of registrant" in (nxt.get("text", "") or "").lower()
            ):
                out.append(span)
        return out
    except Exception:
        return []
