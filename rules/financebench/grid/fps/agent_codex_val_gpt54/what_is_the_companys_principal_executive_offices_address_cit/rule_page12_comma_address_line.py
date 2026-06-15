def rule_page12_comma_address_line(doc: dict) -> list[dict]:
    """Match early cover-page address lines written as comma-separated full addresses."""
    try:
        bad_phrases = (
            "washington, d.c.",
            "exact name of registrant",
            "state or other jurisdiction",
            "i.r.s.",
            "irs ",
            "commission file",
            "telephone number",
            "securities registered",
            "trading symbol",
            "name of each exchange",
            "zip code",
            "annual report",
            "quarterly report",
            "current report",
            "form 10-",
            "form 8-k",
        )

        out = []
        for span in doc.get("texts", [])[:70]:
            if span.get("page_no", 999) > 2:
                continue

            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if not text or len(text) > 120:
                continue
            if any(p in low for p in bad_phrases):
                continue
            if text.count(",") < 2:
                continue
            if not any(ch.isdigit() for ch in text):
                continue

            out.append(span)

        return out
    except Exception:
        return []
