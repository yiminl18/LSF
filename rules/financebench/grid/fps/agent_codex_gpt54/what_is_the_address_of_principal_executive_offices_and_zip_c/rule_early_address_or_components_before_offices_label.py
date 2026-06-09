def rule_early_address_or_components_before_offices_label(doc: dict) -> list[dict]:
    """Match early-page address spans, or split address components, immediately before the principal-offices label."""
    try:
        import re

        street_re = re.compile(
            r"\b(?:street|st\.?|drive|dr\.?|avenue|ave\.?|road|rd\.?|way|plaza|"
            r"boulevard|blvd\.?|lane|ln\.?|suite|ste\.?|parkway|circle|court|"
            r"highway|hwy\.?)\b",
            re.I,
        )
        state_abbrev_re = re.compile(
            r"^(?:AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|"
            r"MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|"
            r"SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)$"
        )
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

        def _clean(text: str) -> str:
            return " ".join((text or "").split())

        def _looks_address(text: str) -> bool:
            cleaned = _clean(text)
            lowered = cleaned.lower()
            if not cleaned or any(marker in lowered for marker in marker_phrases):
                return False
            if street_re.search(cleaned):
                return True
            if "," in cleaned and any(ch.isdigit() for ch in cleaned):
                return True
            if lowered.startswith("one ") and "," in cleaned:
                return True
            return False

        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            label_text = _clean(span.get("text") or "")
            lowered = label_text.lower()
            if (
                span.get("label") == "text"
                and (
                    "principal executive offices are located at" in lowered
                    or "our principal executive offices are located at" in lowered
                )
            ):
                results.append(span)
                continue
            if "address of principal executive offices" not in lowered or "zip code" in lowered:
                continue

            if i - 1 >= 0:
                prev = texts[i - 1]
                if prev.get("page_no") == span.get("page_no") and _looks_address(prev.get("text") or ""):
                    results.append(prev)
                    continue

            components = []
            has_street_line = False
            for j in range(max(0, i - 6), i):
                candidate = texts[j]
                if candidate.get("page_no") != span.get("page_no"):
                    continue

                candidate_text = _clean(candidate.get("text") or "")
                candidate_lowered = candidate_text.lower()
                if not candidate_text or any(marker in candidate_lowered for marker in marker_phrases):
                    continue

                is_component = False
                if _looks_address(candidate_text):
                    is_component = True
                    has_street_line = True
                elif state_abbrev_re.fullmatch(candidate_text.strip()):
                    is_component = True
                elif candidate_text.endswith(",") and len(candidate_text.split()) <= 4:
                    is_component = True

                if is_component:
                    components.append(candidate)

            if has_street_line and len(components) >= 2:
                results.extend(components)

        return results
    except Exception:
        return []
