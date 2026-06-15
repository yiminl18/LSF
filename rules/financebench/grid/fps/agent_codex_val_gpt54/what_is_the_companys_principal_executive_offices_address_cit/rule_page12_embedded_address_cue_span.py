def rule_page12_embedded_address_cue_span(doc: dict) -> list[dict]:
    """Match page-1/2 spans where the address text is embedded in the cue span itself."""
    try:
        out = []
        cue = "(address of principal executive offices)"

        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 2:
                continue
            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if "address of principal executive offices" not in low:
                continue
            if len(text) > len(cue) + 12:
                out.append(span)

        return out
    except Exception:
        return []
