def rule_page12_standalone_address_line(doc: dict) -> list[dict]:
    """Match early standalone page-1/2 spans that look like office address lines."""
    try:
        street_markers = (
            " street", " avenue", " ave ", " ave.", " road", " drive", " plaza",
            " boulevard", " blvd", " park way", " lane", " way", " place",
            " loop", " trail", " terrace", " court", " highway", " hwy ",
        )
        state_terms = (
            "california", "minnesota", "new york", "new jersey", "virginia",
            "washington", "illinois", "maryland", "united kingdom",
        )
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

        out = []
        for span in doc.get("texts", [])[:120]:
            if span.get("page_no", 999) > 2:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue

            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if not text or len(text) < 8 or len(text) > 110:
                continue
            if low.startswith("(") and low.endswith(")"):
                continue
            if any(p in low for p in bad_phrases):
                continue

            padded = f" {low} "
            has_street = any(marker in padded for marker in street_markers)
            has_number_and_comma = "," in text and any(ch.isdigit() for ch in text)
            has_state_location = "," in text and any(term in low for term in state_terms)

            if has_street or has_number_and_comma or has_state_location:
                out.append(span)

        return out
    except Exception:
        return []
