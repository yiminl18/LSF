def rule_exact_name_registrant_block(doc: dict) -> list[dict]:
    """Match spans near the 'Exact name of registrant' cover-page identity block."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            path = (((span.get("structure") or {}).get("path_text")) or "").lower()
            if "exact name of registrant" in text or "exact name of registrant" in path:
                out.append(span)
        return out
    except Exception:
        return []
