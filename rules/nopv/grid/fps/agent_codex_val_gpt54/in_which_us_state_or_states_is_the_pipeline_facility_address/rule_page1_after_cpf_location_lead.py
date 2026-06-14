def rule_page1_after_cpf_location_lead(doc: dict) -> list[dict]:
    """Match the page-1 PHMSA location lead paragraph that appears immediately after a CPF marker."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            if not (
                span.get("page_no") == 1
                and span.get("label") == "text"
                and "phmsa" in lowered
                and ("inspect" in lowered or "investigat" in lowered)
            ):
                continue

            prev_spans = [s for s in texts[max(0, i - 3):i] if s.get("page_no") == 1]
            if any("cpf" in ((s.get("text") or "").lower()) for s in prev_spans):
                out.append(span)
        return out
    except Exception:
        return []
