def rule_page1_inline_address_and_telephone_blob(doc: dict) -> list[dict]:
    """Match page-1 address blobs that also inline the registrant telephone number."""
    try:
        import re

        results = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue

            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            digits = re.sub(r"\D", "", text)
            if len(digits) < 15:
                continue
            if "telephone" not in lowered or "registrant" not in lowered:
                continue
            if "address of principal executive offices" not in lowered:
                continue

            results.append(span)

        return results
    except Exception:
        return []
