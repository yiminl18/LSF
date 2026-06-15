def rule_page16_business_intro_company_sentence(doc: dict) -> list[dict]:
    """Match early business-introduction sentences that restate the company name."""
    try:
        out = []
        intro_phrases = (
            "incorporated under the laws",
            "began operations",
            "and its subsidiaries",
            "is a leading",
            "collectively",
        )
        corp_tokens = (" inc.", " incorporated", " corporation", " company", " plc")

        for span in doc.get("texts", []):
            if (span.get("page_no") or 999) > 6:
                continue
            if span.get("label") != "text":
                continue

            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if len(text) > 900:
                continue
            if not any(phrase in low for phrase in intro_phrases):
                continue
            if not any(token in low for token in corp_tokens):
                continue
            out.append(span)
        return out
    except Exception:
        return []
