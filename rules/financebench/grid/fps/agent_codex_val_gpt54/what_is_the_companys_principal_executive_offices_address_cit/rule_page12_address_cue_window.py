def rule_page12_address_cue_window(doc: dict) -> list[dict]:
    """Capture the local page-1/2 span window around the principal-offices cue."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen = set()

        for i, span in enumerate(texts):
            text = span.get("text") or ""
            if span.get("page_no", 999) > 2:
                continue
            if "address of principal executive offices" not in text.lower():
                continue

            page = span.get("page_no")
            for j in range(max(0, i - 5), min(len(texts), i + 3)):
                near = texts[j]
                if near.get("page_no") != page:
                    continue
                key = id(near)
                if key not in seen:
                    out.append(near)
                    seen.add(key)

        return out
    except Exception:
        return []
