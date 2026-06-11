def rule_near_zip_code_label(doc: dict) -> list[dict]:
    """Match phone-number spans near a zip-code label in the cover-page identity block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for i, span in enumerate(texts):
            text = span.get("text", "") or ""
            if re.search(r"\(Zip Code\)|zip code", text, re.I):
                for j in range(max(0, i - 3), min(len(texts), i + 5)):
                    cand = texts[j]
                    if cand.get("page_no") == span.get("page_no") and phone_re.search(cand.get("text", "") or ""):
                        out.append(cand)
        return out
    except Exception:
        return []
