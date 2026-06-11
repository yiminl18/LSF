def rule_exchange_in_additional_information_section(doc: dict) -> list[dict]:
    """Match narrative exchange mentions in sections like 'Additional Information' or 'Other Information'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "")
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'additional information|other information', path, re.I) and re.search(r'listed on .*?(new york stock exchange|nasdaq)', combined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
