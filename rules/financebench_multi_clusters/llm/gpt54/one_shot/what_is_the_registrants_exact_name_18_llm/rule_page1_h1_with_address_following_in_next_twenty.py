def rule_page1_h1_with_address_following_in_next_twenty(doc: dict) -> list[dict]:
    """Match page-1 H1 spans with an address caption within the next twenty spans."""
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
            for j in range(i + 1, min(i + 21, len(texts))):
                if texts[j].get("page_no") != 1:
                    break
                t = (texts[j].get("text", "") or "").lower()
                if "address of principal executive offices" in t or "address and telephone number" in t:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
