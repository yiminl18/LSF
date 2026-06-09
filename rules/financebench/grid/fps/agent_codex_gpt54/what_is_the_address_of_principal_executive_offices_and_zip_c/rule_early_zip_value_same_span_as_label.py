def rule_early_zip_value_same_span_as_label(doc: dict) -> list[dict]:
    """Match early-page spans that contain both a zip or postcode value and a zip-code label."""
    try:
        import re

        zip_re = re.compile(r"\b\d{5}(?:-\d{4})?\b|\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)

        results = []
        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 2:
                continue

            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            if "zip code" not in lowered or "telephone number" in lowered:
                continue
            if zip_re.search(text):
                results.append(span)

        return results
    except Exception:
        return []
