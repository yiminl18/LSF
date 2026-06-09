def rule_early_value_before_combined_address_zip_label(doc: dict) -> list[dict]:
    """Match early-page full address values that appear immediately before a combined address-and-zip label."""
    try:
        import re

        street_re = re.compile(
            r"\b(?:street|st\.?|drive|dr\.?|avenue|ave\.?|road|rd\.?|way|plaza|"
            r"boulevard|blvd\.?|lane|ln\.?|suite|ste\.?|parkway|circle|court|"
            r"highway|hwy\.?)\b",
            re.I,
        )
        zip_re = re.compile(r"\b\d{5}(?:-\d{4})?\b|\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
        marker_phrases = (
            "jurisdiction",
            "commission file",
            "exact name of registrant",
            "employer identification",
            "i.r.s.",
            "irs employer",
            "telephone number",
            "former name",
            "trading symbol",
            "exchange on which",
            "securities registered",
            "pursuant to section",
        )

        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            label_text = " ".join((span.get("text") or "").split())
            lowered = label_text.lower()
            if "address of principal executive offices" not in lowered or "zip code" not in lowered:
                continue
            if i - 1 < 0:
                continue

            prev = texts[i - 1]
            if prev.get("page_no") != span.get("page_no"):
                continue

            prev_text = " ".join((prev.get("text") or "").split())
            prev_lowered = prev_text.lower()
            if not prev_text or any(marker in prev_lowered for marker in marker_phrases):
                continue
            if street_re.search(prev_text) or (zip_re.search(prev_text) and any(ch.isalpha() for ch in prev_text)):
                results.append(prev)

        return results
    except Exception:
        return []
