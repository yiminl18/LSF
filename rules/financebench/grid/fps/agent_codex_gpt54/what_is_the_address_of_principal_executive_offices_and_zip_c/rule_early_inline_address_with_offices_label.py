def rule_early_inline_address_with_offices_label(doc: dict) -> list[dict]:
    """Match early-page spans that inline the address text before the principal-offices label."""
    try:
        import re

        street_re = re.compile(
            r"\b(?:street|st\.?|drive|dr\.?|avenue|ave\.?|road|rd\.?|way|plaza|"
            r"boulevard|blvd\.?|lane|ln\.?|suite|ste\.?|parkway|circle|court|"
            r"highway|hwy\.?)\b",
            re.I,
        )
        zip_re = re.compile(r"\b\d{5}(?:-\d{4})?\b|\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)

        results = []
        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 2:
                continue

            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            if "address of principal executive offices" not in lowered:
                continue

            before_label = text.split("(Address of principal executive offices)", 1)[0]
            if before_label == text:
                before_label = lowered.split("address of principal executive offices", 1)[0]

            if street_re.search(before_label) or (zip_re.search(before_label) and any(ch.isalpha() for ch in before_label)):
                results.append(span)

        return results
    except Exception:
        return []
