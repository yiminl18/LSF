def rule_page1_phone_after_telephone_label(doc: dict) -> list[dict]:
    """Match page-1 phone-value spans that appear immediately after a telephone label span."""
    try:
        import re

        def _looks_phone_value(text: str) -> bool:
            text = " ".join((text or "").split())
            digits = re.sub(r"\D", "", text)
            if len(digits) < 10 or len(digits) > 15:
                return False
            if re.search(r"[A-Za-z@]", text):
                return False
            if not re.fullmatch(r"[+()0-9.\-\s]+", text):
                return False
            if len(digits) == 10 and text.isdigit():
                return True
            return any(ch in text for ch in "+()- .")

        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts[1:], start=1):
            if span.get("page_no") != 1:
                continue

            text = " ".join((span.get("text") or "").split())
            if not _looks_phone_value(text):
                continue

            prev = texts[i - 1]
            if prev.get("page_no") != 1:
                continue

            prev_text = " ".join((prev.get("text") or "").split()).lower()
            if "telephone" not in prev_text:
                continue

            results.append(span)

        return results
    except Exception:
        return []
