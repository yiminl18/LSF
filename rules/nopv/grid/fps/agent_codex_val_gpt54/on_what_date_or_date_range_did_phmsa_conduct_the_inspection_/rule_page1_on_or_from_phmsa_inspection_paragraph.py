def rule_page1_on_or_from_phmsa_inspection_paragraph(doc: dict) -> list[dict]:
    """Match the page-1 PHMSA lead paragraph that starts with On or From for inspection or OPS investigation notices."""
    try:
        import re

        start_on_or_from_re = re.compile(
            r"^\s*(?:on|from)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)",
            re.I,
        )
        start_from_re = re.compile(
            r"^\s*from\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)",
            re.I,
        )

        out = []
        for span in doc.get("texts", []):
            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            is_inspection_lead = (
                start_on_or_from_re.search(text)
                and "phmsa" in lowered
                and ("inspect" in lowered or "inspection" in lowered)
                and "investigation" not in lowered
            )
            is_ops_investigation_lead = (
                start_from_re.search(text)
                and "phmsa" in lowered
                and "office of pipeline safety" in lowered
                and "investigat" in lowered
            )
            if (
                span.get("page_no") == 1
                and span.get("label") == "text"
                and (is_inspection_lead or is_ops_investigation_lead)
            ):
                out.append(span)
        return out
    except Exception:
        return []
