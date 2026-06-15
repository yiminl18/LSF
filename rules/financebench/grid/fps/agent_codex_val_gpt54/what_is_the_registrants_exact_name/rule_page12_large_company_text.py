def rule_page12_large_company_text(doc: dict) -> list[dict]:
    """Match large standalone company-name text spans on page 1-2."""
    try:
        import re

        bad_phrases = (
            "united states",
            "securities and exchange commission",
            "form 10-",
            "form 8-k",
            "current report",
            "annual report",
            "quarterly report",
            "transition report",
            "commission file",
            "exact name of registrant",
            "address of principal",
            "news release",
        )

        out = []
        for span in doc.get("texts", []):
            if (span.get("page_no") or 999) > 2:
                continue
            if span.get("label") != "text":
                continue
            if (span.get("size") or 0) < 16:
                continue

            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if not text or len(text) > 120 or len(text.split()) > 8:
                continue
            if any(phrase in low for phrase in bad_phrases):
                continue
            if re.fullmatch(r"[\W\d]+", text or ""):
                continue
            if sum(ch.isalpha() for ch in text) < 4:
                continue
            out.append(span)
        return out
    except Exception:
        return []
