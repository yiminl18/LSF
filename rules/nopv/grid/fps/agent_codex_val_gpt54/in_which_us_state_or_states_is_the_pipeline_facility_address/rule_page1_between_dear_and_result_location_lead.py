def rule_page1_between_dear_and_result_location_lead(doc: dict) -> list[dict]:
    """Match the page-1 PHMSA location lead paragraph that falls between Dear and the result lead-in."""
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

            prev_spans = [s for s in texts[max(0, i - 5):i] if s.get("page_no") == 1]
            next_spans = [s for s in texts[i + 1:i + 3] if s.get("page_no") == 1]
            if any("dear " in ((s.get("text") or "").lower()) for s in prev_spans) and any(
                result_re.search(" ".join((s.get("text") or "").split())) for s in next_spans
            ):
                out.append(span)
        return out
    except Exception:
        return []
