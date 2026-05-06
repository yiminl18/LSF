def rule_boeing_other_information_phone(doc: dict) -> list[dict]:
    """Retrieve Boeing's Item 1 Other Information span containing the telephone number."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "text":
                continue
            txt = span.get("text", "") or ""
            path = ((span.get("structure") or {}).get("path_text", "") or "")
            if span.get("page_no") == 6 and "other information" in path.lower() and "telephone number is" in txt.lower():
                out.append(span)
        return out
    except Exception:
        return []

