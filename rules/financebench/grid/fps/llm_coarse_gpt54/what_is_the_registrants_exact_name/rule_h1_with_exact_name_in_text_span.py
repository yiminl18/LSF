def rule_h1_with_exact_name_in_text_span(doc: dict) -> list[dict]:
    """Match H1/section_header spans whose text_span contains the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "section_header":
                continue
            if span.get("structure", {}).get("level") not in {"H1", "H2"}:
                continue
            if "exact name of registrant" in ((span.get("text_span") or "").lower()):
                out.append(span)
        return out
    except Exception:
        return []
