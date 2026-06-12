def rule_profile_of_economy_page_title_and_following(doc: dict) -> list[dict]:
    """Match the Profile of the Economy title and the next several spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if txt == "Profile of the Economy":
                out.append(span)
                for j in range(i + 1, min(i + 10, len(texts))):
                    out.append(texts[j])
        return out
    except Exception:
        return []
