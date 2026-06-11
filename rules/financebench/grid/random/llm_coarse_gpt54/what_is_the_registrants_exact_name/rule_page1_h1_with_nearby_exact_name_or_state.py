def rule_page1_h1_with_nearby_exact_name_or_state(doc: dict) -> list[dict]:
    """Match page-1 H1 headers with nearby exact-name caption or state/incorporation labels."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                nearby = " ".join((s.get("text") or "").lower() for s in texts[max(0, i-1):i+12])
                if (
                    "exact name of registrant as specified in its charter" in nearby
                    or "state or other jurisdiction of incorporation" in nearby
                    or "state or other jurisdiction of incorporation or organization" in nearby
                ):
                    if "FORM 10-" not in (span.get("text") or "").upper():
                        out.append(span)
        return out
    except Exception:
        return []
