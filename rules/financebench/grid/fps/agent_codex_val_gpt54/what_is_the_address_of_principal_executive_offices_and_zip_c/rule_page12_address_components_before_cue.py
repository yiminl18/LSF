def rule_page12_address_components_before_cue(doc: dict) -> list[dict]:
    """Match early page-1/2 address, city, and state spans immediately before the cue."""
    try:
        street_markers = (
            " street", " avenue", " ave ", " ave.", " road", " drive", " plaza",
            " boulevard", " blvd", " park way", " lane", " way", " place",
            " loop", " trail", " terrace", " court", " highway", " hwy ",
        )
        full_states = {
            "california", "minnesota", "new york", "new jersey", "virginia",
            "washington", "illinois", "maryland", "delaware", "united kingdom",
        }
        abbrev_states = {
            "CA", "IL", "MD", "MN", "NJ", "NY", "VA", "WA",
        }
        bad_phrases = (
            "exact name of registrant",
            "state or other jurisdiction",
            "employer identification",
            "commission file",
            "telephone number",
            "trading symbol",
            "exchange on which registered",
            "title of each class",
            "securities registered",
            "registrant had",
            "for the fiscal year",
            "for the quarterly period",
            "for the transition period",
            "washington, d.c. 20549",
        )

        texts = doc.get("texts", [])
        out = []
        seen = set()

        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue
            if "address of principal executive offices" not in ((span.get("text") or "").lower()):
                continue

            page = span.get("page_no")
            found_address_line = False
            for j in range(max(0, i - 6), i):
                prev = texts[j]
                if prev.get("page_no") != page:
                    continue
                text = (prev.get("text") or "").replace("\n", " ").strip()
                low = text.lower()
                if not text or len(text) < 4 or len(text) > 110:
                    continue
                if low.startswith("(") and low.endswith(")"):
                    continue
                if any(p in low for p in bad_phrases):
                    continue

                padded = f" {low} "
                if any(marker in padded for marker in street_markers) or ("," in text and any(ch.isdigit() for ch in text)):
                    found_address_line = True

            if not found_address_line:
                continue

            for j in range(max(0, i - 6), i):
                prev = texts[j]
                if prev.get("page_no") != page:
                    continue
                text = (prev.get("text") or "").replace("\n", " ").strip()
                low = text.lower()
                token = text.strip().strip(",")
                if not text or len(text) < 2 or len(text) > 110:
                    continue
                if low.startswith("(") and low.endswith(")"):
                    continue
                if any(p in low for p in bad_phrases):
                    continue

                padded = f" {low} "
                is_address_line = any(marker in padded for marker in street_markers) or ("," in text and any(ch.isdigit() for ch in text))
                is_city_line = text.endswith(",") and len(text) <= 50 and any(ch.isalpha() for ch in text)
                is_state_token = (token in abbrev_states and token.isupper()) or (low in full_states)

                if is_address_line or is_city_line or is_state_token:
                    key = id(prev)
                    if key not in seen:
                        out.append(prev)
                        seen.add(key)

        return out
    except Exception:
        return []
