def rule_page1_compact_inline_registrant_telephone(doc: dict) -> list[dict]:
    """Match page-1 spans that inline the registrant-telephone label with the phone value."""
    try:
        import re

        results = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue

            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            digits = re.sub(r"\D", "", text)
            if len(digits) < 10:
                continue
            if "telephone" not in lowered or "registrant" not in lowered:
                continue
            if "address of principal executive offices" in lowered:
                continue

            results.append(span)

        return results
    except Exception:
        return []
