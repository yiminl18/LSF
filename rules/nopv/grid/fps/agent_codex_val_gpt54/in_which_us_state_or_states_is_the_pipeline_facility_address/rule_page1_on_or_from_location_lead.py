def rule_page1_on_or_from_location_lead(doc: dict) -> list[dict]:
    """Match early PHMSA location lead paragraphs that start with On, From, or Following and describe an inspection or investigation."""
    try:
        import re

        start_re = re.compile(
            r"^\s*(?:\d+\s+)?(?:on|from|following)\s+(?:the\s+)?(?:(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|[A-Za-z])",
            re.I,
        )

        out = []
        for span in doc.get("texts", []):
            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            if (
                span.get("page_no", 0) <= 2
                and span.get("label") == "text"
                and start_re.search(text)
                and "phmsa" in lowered
                and ("inspect" in lowered or "investigat" in lowered)
                and "as a result of" not in lowered
            ):
                out.append(span)
        return out
    except Exception:
        return []
