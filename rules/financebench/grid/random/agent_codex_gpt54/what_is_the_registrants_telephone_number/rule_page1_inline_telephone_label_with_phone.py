def rule_page1_inline_telephone_label_with_phone(doc: dict) -> list[dict]:
    """Match page-1 cover spans that include both the telephone label and the phone number."""
    try:
        import re

        phone_re = re.compile(r"(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}")

        hits: list[dict] = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            low = text.lower()
            if span.get("page_no") != 1 or span.get("label") == "table":
                continue
            if "telephone number" not in low or "area code" not in low:
                continue
            if phone_re.search(text):
                hits.append(span)
        return hits
    except Exception:
        return []
