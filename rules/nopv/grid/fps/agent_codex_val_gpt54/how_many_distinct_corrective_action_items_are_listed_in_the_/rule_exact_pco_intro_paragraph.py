def rule_exact_pco_intro_paragraph(doc: dict) -> list[dict]:
    """Match the introductory paragraph under an exact Proposed Compliance Order path."""
    try:
        texts = doc.get("texts", [])
        exact_items = [
            s for s in texts
            if s.get("label") == "list_item"
            and (((s.get("structure") or {}).get("path_text") or "").strip().upper() == "PROPOSED COMPLIANCE ORDER")
        ]
        if not exact_items:
            return []
        for span in texts:
            if (
                span.get("label") == "text"
                and (((span.get("structure") or {}).get("path_text") or "").strip().upper() == "PROPOSED COMPLIANCE ORDER")
            ):
                return [span]
        return []
    except Exception:
        return []
