def rule_page12_country_address_line(doc: dict) -> list[dict]:
    """Match early cover-page address lines that end with a country-style location string."""
    try:
        out = []

        for span in doc.get("texts", [])[:70]:
            if span.get("page_no", 999) > 2:
                continue

            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if not text or len(text) > 140:
                continue
            if not any(ch.isdigit() for ch in text):
                continue
            if "united kingdom" in low or low.endswith(" jersey") or ", jersey" in low:
                out.append(span)

        return out
    except Exception:
        return []
