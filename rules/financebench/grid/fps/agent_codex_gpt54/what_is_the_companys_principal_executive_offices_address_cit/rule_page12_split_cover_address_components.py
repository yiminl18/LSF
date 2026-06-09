def rule_page12_split_cover_address_components(doc: dict) -> list[dict]:
    """Match multi-line early-page address components that sit immediately before the offices label."""
    try:
        import re

        street_re = re.compile(
            r"\b(?:street|st\.?|drive|dr\.?|avenue|ave\.?|road|rd\.?|way|plaza|"
            r"boulevard|blvd\.?|lane|ln\.?|suite|ste\.?|park|tower)\b",
            re.I,
        )
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
            if "address of principal executive offices" not in lowered:
                continue
            if len(lowered) > 90 or street_re.search(label_text) or "," in label_text:
                continue

            components = []
            for j in range(max(0, i - 6), i):
                candidate = texts[j]
                if candidate.get("page_no") != span.get("page_no"):
                    continue

                candidate_text = " ".join((candidate.get("text") or "").split())
                candidate_lowered = candidate_text.lower()
                if not candidate_text or len(candidate_text) > 120:
                    continue
                if any(
                    marker in candidate_lowered
                    for marker in (
                        "jurisdiction",
                        "commission file",
                        "exact name of registrant",
                        "employer identification",
                        "i.r.s.",
                        "irs employer",
                        "zip code",
                        "telephone number",
                        "former name",
                        "trading symbol",
                        "exchange on which",
                        "securities registered",
                        "pursuant to section",
                    )
                ):
                    continue
                if not (
                    street_re.search(candidate_text)
                    or "," in candidate_text
                    or state_abbrev_re.fullmatch(candidate_text.strip())
                ):
                    continue

                components.append(candidate)

            if len(components) >= 2:
                results.extend(components)

        return results
    except Exception:
        return []
