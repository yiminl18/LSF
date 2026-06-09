def rule_phone_with_principal_executive_offices_context(doc: dict) -> list[dict]:
    """Match phone spans tied to the principal executive offices cover caption or sentence."""
    try:
        import re

        phone_re = re.compile(r"(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}")

        texts = doc.get("texts", [])
        hits: list[dict] = []
        for idx, span in enumerate(texts):
            text = (span.get("text") or "").strip()
            low = text.lower()
            if span.get("label") == "table" or not phone_re.search(text):
                continue
            next_text = " ".join((texts[j].get("text") or "") for j in range(idx + 1, min(len(texts), idx + 3))).lower()
            if "principal executive offices" in next_text and "address and telephone number" in next_text:
                hits.append(span)
                continue
            if "principal executive offices" in low and "telephone number" in low:
                hits.append(span)
        return hits
    except Exception:
        return []
