def rule_page1_h1_with_state_following_in_next_ten(doc: dict) -> list[dict]:
    """Match page-1 H1 spans with a state-of-incorporation caption within the next ten spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                continue
            for j in range(i + 1, min(i + 11, len(texts))):
                if texts[j].get("page_no") != 1:
                    break
                t = (texts[j].get("text", "") or "").lower()
                if "state or other jurisdiction of incorporation" in t or "state or other jurisdiction of incorporation or organization" in t:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
