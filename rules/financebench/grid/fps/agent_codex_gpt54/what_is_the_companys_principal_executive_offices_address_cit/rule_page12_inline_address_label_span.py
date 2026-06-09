def rule_page12_inline_address_label_span(doc: dict) -> list[dict]:
    """Match early-page spans that contain both the address text and the principal-offices label."""
    try:
        import re

        street_re = re.compile(
            r"\b(?:street|st\.?|drive|dr\.?|avenue|ave\.?|road|rd\.?|way|plaza|"
            r"boulevard|blvd\.?|lane|ln\.?|suite|ste\.?|park|tower)\b",
            re.I,
        )

        results = []
        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 2:
                continue

            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            if "address of principal executive offices" not in lowered:
                continue

            before_label = lowered.split("address of principal executive offices", 1)[0]
            if street_re.search(text) or "," in text or re.search(r"\d", before_label):
                results.append(span)

        return results
    except Exception:
        return []
