def rule_page1_phone_before_registrant_telephone_label(doc: dict) -> list[dict]:
    """Match page-1 phone-value spans that are immediately followed by the registrant-telephone label."""
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
        for i, span in enumerate(texts[:-1]):
            if span.get("page_no") != 1:
                continue

            text = " ".join((span.get("text") or "").split())
            if not _looks_phone_value(text):
                continue

            nxt = texts[i + 1]
            if nxt.get("page_no") != 1:
                continue

            nxt_text = " ".join((nxt.get("text") or "").split())
            nxt_lower = nxt_text.lower()
            if "telephone" not in nxt_lower or "registrant" not in nxt_lower:
                continue

            if re.sub(r"\D", "", text) and re.sub(r"\D", "", text) in re.sub(r"\D", "", nxt_text):
                continue

            results.append(span)

        return results
    except Exception:
        return []
