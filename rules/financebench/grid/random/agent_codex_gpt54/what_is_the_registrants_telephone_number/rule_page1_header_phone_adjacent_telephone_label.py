def rule_page1_header_phone_adjacent_telephone_label(doc: dict) -> list[dict]:
    """Match page-1 phone-valued headers with the telephone label in an adjacent span."""
    try:
        import re

        phone_re = re.compile(r"^(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}$")
        label_re = re.compile(r"telephone number.*area code", re.IGNORECASE)

        texts = doc.get("texts", [])
        hits: list[dict] = []
        for idx, span in enumerate(texts):
            text = (span.get("text") or "").strip()
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            if not phone_re.fullmatch(text):
                continue
            prev_text = " ".join((texts[j].get("text") or "") for j in range(max(0, idx - 2), idx))
            next_text = " ".join((texts[j].get("text") or "") for j in range(idx + 1, min(len(texts), idx + 3)))
            if label_re.search(prev_text) or label_re.search(next_text):
                hits.append(span)
        return hits
    except Exception:
        return []
