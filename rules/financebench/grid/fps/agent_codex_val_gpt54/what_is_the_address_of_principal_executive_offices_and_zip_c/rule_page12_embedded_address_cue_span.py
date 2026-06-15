def rule_page12_embedded_address_cue_span(doc: dict) -> list[dict]:
    """Match early page-1/2 spans where the address text is embedded in the cue span."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 2:
                continue

            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if "address of principal executive offices and zip code" in low:
                out.append(span)
            elif "address of principal executive offices" in low and len(text) > 45:
                out.append(span)

        return out
    except Exception:
        return []
