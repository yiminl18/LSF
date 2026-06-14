def rule_page1_before_result_location_lead(doc: dict) -> list[dict]:
    """Match the page-1 PHMSA location lead paragraph that is followed by an As a result lead-in."""
    try:
        import re

        result_re = re.compile(r"^\s*as a result of the (?:on-site |field )?(?:inspection|investigation)", re.I)

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

            next_spans = [s for s in texts[i + 1:i + 3] if s.get("page_no") == 1]
            if any(result_re.search(" ".join((s.get("text") or "").split())) for s in next_spans):
                out.append(span)
        return out
    except Exception:
        return []
