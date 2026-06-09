def rule_early_value_before_zip_label(doc: dict) -> list[dict]:
    """Match early-page zip or postcode values that appear immediately before a zip-code label span."""
    try:
        import re

        zip_re = re.compile(r"\b\d{5}(?:-\d{4})?\b|\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
        state_abbrev_re = re.compile(
            r"^(?:AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|"
            r"MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|"
            r"SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)$"
        )

        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            label_text = " ".join((span.get("text") or "").split())
            lowered = label_text.lower()
            if "zip code" not in lowered or zip_re.search(label_text):
                continue
            if i - 1 < 0:
                continue

            prev = texts[i - 1]
            if prev.get("page_no") != span.get("page_no"):
                continue

            prev_text = " ".join((prev.get("text") or "").split())
            if not prev_text or state_abbrev_re.fullmatch(prev_text.strip()):
                continue
            if zip_re.search(prev_text) and len(prev_text) <= 40:
                results.append(prev)

        return results
    except Exception:
        return []
