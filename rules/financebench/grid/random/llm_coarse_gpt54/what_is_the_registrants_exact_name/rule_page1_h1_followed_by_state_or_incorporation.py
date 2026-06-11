def rule_page1_h1_followed_by_state_or_incorporation(doc: dict) -> list[dict]:
    """Match page-1 H1 headers followed nearby by state/incorporation or EIN labels."""
    try:
        texts = doc.get("texts", [])
        out = []
        keys = [
            "state or other jurisdiction of incorporation",
            "state or other jurisdiction of incorporation or organization",
            "i.r.s. employer identification no.",
            "commission file no.",
        ]
        for i, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                window = texts[i + 1:i + 12]
                joined = " ".join((w.get("text") or "").lower() for w in window)
                if any(k in joined for k in keys):
                    if "FORM 10-" not in (span.get("text") or "").upper():
                        out.append(span)
        return out
    except Exception:
        return []
